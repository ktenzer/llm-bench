#!/usr/bin/env python3
"""
QPS-based benchmark script for LLM endpoint.

This script sends requests at a specified QPS (queries per second) rate
and measures when the endpoint starts failing (503s, timeouts, etc.).

Installation:
    pip install aiohttp tqdm
    pip install matplotlib  # Optional, for visualization

Usage:
    # Basic benchmark (P50, P95, P99 by default)
    python benchmark.py --endpoint https://api.together.xyz/v1/chat/completions \\
                        --api-key YOUR_API_KEY \\
                        --model YOUR_MODEL \\
                        --dataset conversations.json \\
                        --qps 1,5,10,20,50 \\
                        --duration 60
    
    # With visualization
    python benchmark.py --endpoint YOUR_ENDPOINT \\
                        --api-key YOUR_KEY \\
                        --model YOUR_MODEL \\
                        --dataset conversations.json \\
                        --qps 1,5,10,20,50 \\
                        --duration 60 \\
                        --plot
    
    # Include tokens/sec metrics
    python benchmark.py --endpoint YOUR_ENDPOINT \\
                        --api-key YOUR_KEY \\
                        --model YOUR_MODEL \\
                        --dataset conversations.json \\
                        --qps 1,5,10 \\
                        --show-tokens
    
    # Show only P50 (compact output)
    python benchmark.py --endpoint YOUR_ENDPOINT \\
                        --api-key YOUR_KEY \\
                        --model YOUR_MODEL \\
                        --dataset conversations.json \\
                        --qps 1,5,10,20 \\
                        --percentiles 50
    
    # Help
    python benchmark.py --help
"""

import argparse
import asyncio
import json
import time
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics

try:
    import aiohttp
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "aiohttp", "tqdm"])
    import aiohttp
    from tqdm import tqdm

# Matplotlib is optional for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    success: bool
    ttft: Optional[float] = None
    total_time: Optional[float] = None
    status_code: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class QPSResults:
    """Results for a QPS test run."""
    qps_target: float
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    
    e2e_latency_p50: float = 0.0
    e2e_latency_p90: float = 0.0
    e2e_latency_p95: float = 0.0
    e2e_latency_p99: float = 0.0
    
    tokens_per_sec_p50: float = 0.0
    tokens_per_sec_p90: float = 0.0
    tokens_per_sec_p95: float = 0.0
    tokens_per_sec_p99: float = 0.0
    
    actual_qps: float = 0.0
    success_rate: float = 0.0
    actual_duration: float = 0.0  # Actual time taken for the test
    
    # Cost calculation fields (optional)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    sustained_tokens_per_sec: float = 0.0  # Overall throughput
    gpu_cost: float = 0.0
    cost_per_million_tokens: float = 0.0
    
    error_counts: Dict[str, int] = field(default_factory=dict)
    status_codes: Dict[int, int] = field(default_factory=dict)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the Duolingo conversation dataset."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversation samples")
    return data


def prepare_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a sample into chat messages format."""
    messages = []
    
    if sample.get('prompt_content'):
        messages.append({
            "role": "system",
            "content": sample['prompt_content']
        })
    
    for turn in sample.get('cut_chat_history', []):
        role = turn.get('role')
        content = turn.get('content')
        if role == 'system':
            continue
        if role and content:
            messages.append({
                "role": role,
                "content": content
            })
    
    return messages


def estimate_input_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate input tokens from messages (rough approximation: chars/4)."""
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
    return max(1, total_chars // 4)


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> RequestMetrics:
    """Send a single streaming request and measure metrics."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    
    start_time = time.time()
    ttft = None
    output_tokens = None  # Will be set from API usage data ONLY
    input_tokens = None   # Will be set from API usage data ONLY
    status_code = None
    chunk_count = 0  # Count chunks for TTFT only, not for token counting
    
    try:
        async with session.post(endpoint, headers=headers, json=payload, 
                               timeout=aiohttp.ClientTimeout(total=60)) as response:
            status_code = response.status
            
            if response.status != 200:
                error_text = await response.text()
                return RequestMetrics(
                    success=False,
                    status_code=status_code,
                    error=f"HTTP {status_code}",
                    timestamp=start_time,
                    total_time=time.time() - start_time
                )
            
            # Process streaming response
            first_token_received = False
            async for line in response.content:
                if not line:
                    continue
                
                line = line.decode('utf-8').strip()
                if not line.startswith('data: '):
                    continue
                
                line = line[6:]
                if line == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(line)
                    if not chunk:
                        continue
                    
                    # Check for usage information - ONLY source of token counts!
                    if 'usage' in chunk and chunk['usage'] is not None:
                        if 'prompt_tokens' in chunk['usage']:
                            input_tokens = chunk['usage']['prompt_tokens']
                        if 'completion_tokens' in chunk['usage']:
                            output_tokens = chunk['usage']['completion_tokens']
                    
                    if not first_token_received:
                        ttft = time.time() - start_time
                        first_token_received = True
                    
                    # Count chunks for debugging, but DON'T use for token counting
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if delta and 'content' in delta and delta['content']:
                            chunk_count += 1
                
                except json.JSONDecodeError:
                    continue
            
            total_time = time.time() - start_time
            
            # DO NOT estimate tokens - only use actual API-reported values
            # If API doesn't provide token counts, leave as None
            
            return RequestMetrics(
                success=True,
                ttft=ttft,
                total_time=total_time,
                status_code=200,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                timestamp=start_time
            )
    
    except asyncio.TimeoutError:
        return RequestMetrics(
            success=False, 
            error="Timeout",
            timestamp=start_time,
            total_time=time.time() - start_time
        )
    except Exception as e:
        return RequestMetrics(
            success=False, 
            error=str(e)[:100],
            timestamp=start_time,
            total_time=time.time() - start_time
        )


async def run_qps_test(
    endpoint: str,
    api_key: str,
    model: str,
    dataset: List[Dict[str, Any]],
    qps_target: float,
    duration: int = 60,
    max_tokens: int = 512,
    temperature: float = 0.7,
    gpu_rate: Optional[float] = None,
    num_gpus: Optional[int] = None,
) -> QPSResults:
    """
    Run QPS test - send requests at specified rate for given duration.
    Optionally calculates cost per million tokens if gpu_rate and num_gpus are provided.
    """
    
    print(f"\n{'='*80}")
    print(f"Running QPS test: {qps_target} queries/second for {duration} seconds")
    print(f"{'='*80}")
    
    # Prepare all request data
    request_data = []
    for sample in dataset:
        messages = prepare_messages(sample)
        if messages:
            request_data.append(messages)
    
    if not request_data:
        raise ValueError("No valid request data prepared")
    
    # Calculate interval between requests
    interval = 1.0 / qps_target
    total_expected_requests = int(qps_target * duration)
    
    print(f"Target: {total_expected_requests} requests ({qps_target} per second)")
    print(f"Interval: {interval:.3f} seconds between requests")
    print(f"Starting test...\n")
    
    results = []
    request_count = 0
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(total=total_expected_requests, desc=f"QPS {qps_target}")
    
    async with aiohttp.ClientSession() as session:
        # Send requests at the target rate
        next_request_time = start_time
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Wait until it's time for the next request
            if current_time < next_request_time:
                await asyncio.sleep(next_request_time - current_time)
            
            # Get messages for this request (cycle through dataset)
            messages = request_data[request_count % len(request_data)]
            
            # Send request (fire and forget - don't wait for completion)
            task = asyncio.create_task(
                send_request(session, endpoint, api_key, model, messages, max_tokens, temperature)
            )
            
            # Store the task to collect results later
            results.append(task)
            request_count += 1
            pbar.update(1)
            
            # Schedule next request
            next_request_time += interval
        
        # Wait for all pending requests to complete
        print("\n\nWaiting for pending requests to complete...")
        completed_results = await asyncio.gather(*results, return_exceptions=True)
    
    pbar.close()
    
    end_time = time.time()
    actual_duration = end_time - start_time
    actual_qps = request_count / actual_duration
    
    # Filter out exceptions
    valid_results = [r for r in completed_results if isinstance(r, RequestMetrics)]
    
    # Calculate statistics
    successful_results = [r for r in valid_results if r.success]
    failed_results = [r for r in valid_results if not r.success]
    
    ttft_values = [r.ttft for r in successful_results if r.ttft is not None]
    e2e_values = [r.total_time for r in successful_results if r.total_time is not None]
    
    # Calculate tokens per second for each successful request
    tokens_per_sec_values = []
    for r in successful_results:
        if r.output_tokens and r.total_time and r.total_time > 0:
            tps = r.output_tokens / r.total_time
            tokens_per_sec_values.append(tps)
    
    # Count errors and status codes
    error_counts = defaultdict(int)
    for r in failed_results:
        error_counts[r.error or "Unknown"] += 1
    
    status_counts = defaultdict(int)
    for r in valid_results:
        if r.status_code:
            status_counts[r.status_code] += 1
    
    # Calculate percentiles
    def safe_percentile(values, percentile):
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    results_obj = QPSResults(
        qps_target=qps_target,
        duration=actual_duration,
        total_requests=len(valid_results),
        successful_requests=len(successful_results),
        failed_requests=len(failed_results),
        actual_qps=actual_qps,
        success_rate=(len(successful_results) / len(valid_results) * 100) if valid_results else 0,
        error_counts=dict(error_counts),
        status_codes=dict(status_counts),
    )
    
    if ttft_values:
        results_obj.ttft_p50 = safe_percentile(ttft_values, 50)
        results_obj.ttft_p90 = safe_percentile(ttft_values, 90)
        results_obj.ttft_p95 = safe_percentile(ttft_values, 95)
        results_obj.ttft_p99 = safe_percentile(ttft_values, 99)
    
    if e2e_values:
        results_obj.e2e_latency_p50 = safe_percentile(e2e_values, 50)
        results_obj.e2e_latency_p90 = safe_percentile(e2e_values, 90)
        results_obj.e2e_latency_p95 = safe_percentile(e2e_values, 95)
        results_obj.e2e_latency_p99 = safe_percentile(e2e_values, 99)
    
    if tokens_per_sec_values:
        results_obj.tokens_per_sec_p50 = safe_percentile(tokens_per_sec_values, 50)
        results_obj.tokens_per_sec_p90 = safe_percentile(tokens_per_sec_values, 90)
        results_obj.tokens_per_sec_p95 = safe_percentile(tokens_per_sec_values, 95)
        results_obj.tokens_per_sec_p99 = safe_percentile(tokens_per_sec_values, 99)
    
    # Store actual duration
    results_obj.actual_duration = actual_duration
    
    # Calculate cost per million tokens if GPU parameters provided
    if gpu_rate is not None and num_gpus is not None:
        # Count total tokens (input + output) from successful requests that have token data
        # ONLY count requests where API provided actual token counts
        requests_with_tokens = [r for r in successful_results 
                                if r.input_tokens is not None and r.output_tokens is not None]
        
        total_input = sum(r.input_tokens for r in requests_with_tokens)
        total_output = sum(r.output_tokens for r in requests_with_tokens)
        total_tokens = total_input + total_output
        
        print(f"\n📊 Token counting: {len(requests_with_tokens)}/{len(successful_results)} "
              f"requests had API-reported token counts")
        
        if len(requests_with_tokens) < len(successful_results):
            missing_pct = ((len(successful_results) - len(requests_with_tokens)) / 
                          len(successful_results) * 100)
            print(f"   ⚠️  WARNING: {missing_pct:.1f}% of requests missing token data!")
            print(f"   Cost calculation based only on requests with actual API token counts")
        
        if total_tokens > 0 and len(requests_with_tokens) > 0:
            # For cost calculation: total tokens processed / time
            # This is the AVERAGE rate at which tokens flowed through the GPU
            # NOT the same as GPU's max processing capability!
            #
            # Example: 3.6M input + 28k output in 30s = 122k avg tokens/sec
            # This is high because input (prefill) is fast, output (decode) is slow
            # The GPU spends most actual compute time on the decode phase
            
            total_tokens_per_second = total_tokens / actual_duration
            
            print(f"   ✅ Tokens (API-reported): {total_input:,} input + {total_output:,} output = {total_tokens:,}")
            print(f"   ⏱️  Test duration: {actual_duration:.1f} seconds")
            
            # Calculate cost per million tokens
            # Formula: (GPU_rate * num_gpus * duration_hours) * (1M / total_tokens)
            # This gives: cost to process the tokens we actually processed, scaled to 1M
            
            duration_hours = actual_duration / 3600.0
            cost_for_this_run = num_gpus * gpu_rate * duration_hours
            cost_per_million = (1_000_000 / total_tokens) * cost_for_this_run
            
            print(f"   💰 Cost formula: (${gpu_rate}/hr × {num_gpus} GPU × {duration_hours:.4f}hr) × (1M / {total_tokens:,})")
            print(f"   💰 Cost for this run: ${cost_for_this_run:.4f}")
            print(f"   💰 Scaled to 1M tokens: ${cost_per_million:.4f}")
            
            # Also track output-only throughput for system capacity metrics
            output_tokens_per_second = total_output / actual_duration if total_output > 0 else 0
            
            # Store detailed cost analysis
            results_obj.total_input_tokens = total_input
            results_obj.total_output_tokens = total_output
            results_obj.total_tokens = total_tokens
            results_obj.sustained_tokens_per_sec = output_tokens_per_second  # Output only for capacity planning
            results_obj.gpu_cost = cost_for_this_run  # Cost for this run
            results_obj.cost_per_million_tokens = cost_per_million
    
    return results_obj


def print_results_table(all_results: List[QPSResults], percentiles: List[int] = [50, 95, 99], show_cost: bool = False):
    """Print results in a formatted table with selected percentiles. Always includes tokens/sec."""
    
    # Build header dynamically based on selected percentiles
    base_cols = [
        ('QPS Target', 12),
        ('Actual QPS', 12),
        ('Success%', 10),
    ]
    
    percentile_cols = []
    for p in percentiles:
        percentile_cols.append((f'TTFT P{p}(s)', 12))
    for p in percentiles:
        percentile_cols.append((f'E2E P{p}(s)', 12))
    # Always show tokens/sec
    for p in percentiles:
        percentile_cols.append((f'Tok/s P{p}', 12))
    
    # Add cost column if cost was calculated
    if show_cost:
        percentile_cols.append(('$/M Tokens', 12))
    
    all_cols = base_cols + percentile_cols
    total_width = sum(width for _, width in all_cols) + len(all_cols) - 1
    
    print("\n" + "="*total_width)
    print("QPS BENCHMARK RESULTS")
    print("="*total_width)
    
    # Print header
    header_parts = [name.ljust(width) for name, width in all_cols]
    print(" ".join(header_parts))
    print("-" * total_width)
    
    # Data rows
    for result in all_results:
        row_parts = []
        
        # Base columns
        row_parts.append(f"{result.qps_target:.1f}".ljust(12))
        row_parts.append(f"{result.actual_qps:.2f}".ljust(12))
        row_parts.append(f"{result.success_rate:.1f}".ljust(10))
        
        # TTFT percentiles
        for p in percentiles:
            val = getattr(result, f'ttft_p{p}', 0.0)
            row_parts.append(f"{val:.3f}".ljust(12))
        
        # E2E percentiles
        for p in percentiles:
            val = getattr(result, f'e2e_latency_p{p}', 0.0)
            row_parts.append(f"{val:.3f}".ljust(12))
        
        # Tokens/sec percentiles - always show
        for p in percentiles:
            val = getattr(result, f'tokens_per_sec_p{p}', 0.0)
            row_parts.append(f"{val:.1f}".ljust(12))
        
        # Cost per million tokens (if calculated)
        if show_cost:
            cost = result.cost_per_million_tokens
            if cost > 0:
                row_parts.append(f"${cost:.4f}".ljust(12))
            else:
                row_parts.append("N/A".ljust(12))
        
        print(" ".join(row_parts))
    
    print("="*total_width)
    
    # Print cost analysis context if cost was calculated
    if any(r.cost_per_million_tokens is not None and r.cost_per_million_tokens > 0 for r in all_results):
        print("\n" + "="*total_width)
        print("COST ANALYSIS NOTES".center(total_width))
        print("="*total_width)
        
        for result in all_results:
            if result.cost_per_million_tokens and result.cost_per_million_tokens > 0:
                # Output tokens/sec for capacity planning
                output_tokens_per_sec = result.sustained_tokens_per_sec
                
                # Calculate GPU utilization estimate based on output generation
                # Rough estimate: H100 can do ~10k-15k output tokens/sec at full utilization for 8B models
                estimated_max_throughput = 12000  # Conservative estimate for 8B model
                gpu_utilization = min(100, (output_tokens_per_sec / estimated_max_throughput) * 100)
                
                # Show cost calculation breakdown
                duration_hours = result.actual_duration / 3600.0
                
                print(f"\nQPS {result.qps_target} ({result.success_rate:.1f}% success):")
                # Calculate input/output ratio
                input_output_ratio = (result.total_input_tokens / result.total_output_tokens 
                                     if result.total_output_tokens > 0 else 0)
                
                print(f"  📊 Tokens (API-reported):")
                print(f"     - Input:  {result.total_input_tokens:,}")
                print(f"     - Output: {result.total_output_tokens:,}")
                print(f"     - Total:  {result.total_tokens:,}")
                print(f"     - Ratio:  {input_output_ratio:.1f}:1 (input:output)")
                if input_output_ratio > 50:
                    print(f"     ⚠️  Very high input/output ratio - most tokens are context (prefill)")
                print(f"  📊 Performance:")
                print(f"     - Output generation rate: {output_tokens_per_sec:,.1f} tokens/sec")
                print(f"     - Est. GPU utilization: ~{gpu_utilization:.0f}% (based on decode phase)")
                print(f"  💰 Cost Breakdown:")
                print(f"     - Test duration: {result.actual_duration:.1f}s ({duration_hours:.4f} hours)")
                print(f"     - GPU cost for test: ${result.gpu_cost:.4f}")
                print(f"     - Cost per 1M tokens: ${result.cost_per_million_tokens:.4f}")
                print(f"     - Formula: (${result.gpu_cost:.4f} ÷ {result.total_tokens:,} tokens) × 1,000,000")
                
                # Add efficiency notes based on utilization
                if gpu_utilization < 20 and result.success_rate >= 99:
                    print(f"  💡 Low output rate ({output_tokens_per_sec:.0f} tok/s) - GPU may be underutilized.")
                    print(f"     Consider: increase QPS or use smaller batch/context sizes.")
                
                # Add warnings
                if result.success_rate < 95:
                    print(f"  ⚠️  WARNING: {result.success_rate:.1f}% success rate - system may be saturated!")
                    print(f"      Cost assumes you can sustain {total_tokens_per_sec:,.0f} tok/s, but errors indicate")
                    print(f"      you'd need MORE GPUs to reliably process at this rate.")
                elif result.success_rate < 99:
                    print(f"  ⚠️  Note: {result.success_rate:.1f}% success rate - some request failures")
        
        print("\n" + "="*total_width)
        print("💡 KEY INSIGHTS:")
        print("    • Cost per token DECREASES as QPS increases (better GPU utilization)")
        print("    • Low QPS = GPU underutilized = higher cost per token")
        print("    • High QPS = GPU busy processing batches = lower cost per token")
        print("    • For production: Use HIGHEST QPS with 99%+ success rate")
        print("\n    📊 Cost Calculation:")
        print("       Formula: (GPU_cost_for_run ÷ total_tokens_in_run) × 1,000,000")
        print("       Tokens: API-reported ONLY (no estimates)")
        print("       Total = input + output from API 'usage' field")
        print("="*total_width)
    
    # Print detailed error information
    for result in all_results:
        if result.failed_requests > 0:
            print(f"\nQPS {result.qps_target} - Errors ({result.failed_requests} total):")
            for error, count in sorted(result.error_counts.items(), key=lambda x: -x[1]):
                print(f"  - {error}: {count} occurrences")
            
            if result.status_codes:
                print(f"  Status codes: {dict(result.status_codes)}")


def create_results_directory() -> str:
    """Create a timestamped results directory and return the path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_results(all_results: List[QPSResults], output_file: str):
    """Save results to JSON file."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results_dict = []
    for result in all_results:
        results_dict.append({
            "qps_target": result.qps_target,
            "actual_qps": result.actual_qps,
            "duration": result.duration,
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "success_rate": result.success_rate,
            "ttft": {
                "p50": result.ttft_p50,
                "p90": result.ttft_p90,
                "p95": result.ttft_p95,
                "p99": result.ttft_p99,
            },
            "e2e_latency": {
                "p50": result.e2e_latency_p50,
                "p90": result.e2e_latency_p90,
                "p95": result.e2e_latency_p95,
                "p99": result.e2e_latency_p99,
            },
            "tokens_per_sec": {
                "p50": result.tokens_per_sec_p50,
                "p90": result.tokens_per_sec_p90,
                "p95": result.tokens_per_sec_p95,
                "p99": result.tokens_per_sec_p99,
            },
            "cost_analysis": {
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "total_tokens": result.total_tokens,
                "sustained_tokens_per_sec": result.sustained_tokens_per_sec,
                "gpu_cost": result.gpu_cost,
                "cost_per_million_tokens": result.cost_per_million_tokens,
            } if result.cost_per_million_tokens > 0 else None,
            "actual_duration": result.actual_duration,
            "errors": result.error_counts,
            "status_codes": result.status_codes,
        })
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


def plot_results(all_results: List[QPSResults], output_file: str = "benchmark_qps.png"):
    """Generate visualization of QPS benchmark results. Always shows TTFT, E2E, and Tokens/sec."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib not installed. Install with: pip install matplotlib")
        print("   Skipping visualization...")
        return
    
    if not all_results:
        print("No results to plot")
        return
    
    # Extract data
    qps_targets = [r.qps_target for r in all_results]
    actual_qps = [r.actual_qps for r in all_results]
    success_rates = [r.success_rate for r in all_results]
    ttft_p50 = [r.ttft_p50 for r in all_results]
    ttft_p90 = [r.ttft_p90 for r in all_results]
    ttft_p95 = [r.ttft_p95 for r in all_results]
    ttft_p99 = [r.ttft_p99 for r in all_results]
    e2e_p50 = [r.e2e_latency_p50 for r in all_results]
    e2e_p90 = [r.e2e_latency_p90 for r in all_results]
    e2e_p95 = [r.e2e_latency_p95 for r in all_results]
    e2e_p99 = [r.e2e_latency_p99 for r in all_results]
    
    # Extract tokens/sec data - always calculate
    tps_p50 = [r.tokens_per_sec_p50 for r in all_results]
    tps_p90 = [r.tokens_per_sec_p90 for r in all_results]
    tps_p95 = [r.tokens_per_sec_p95 for r in all_results]
    tps_p99 = [r.tokens_per_sec_p99 for r in all_results]
    
    # Extract aggregate sustained throughput (output tokens/sec across all requests)
    sustained_throughput = [r.sustained_tokens_per_sec for r in all_results]
    
    # Check if cost data is available
    has_cost = any(r.cost_per_million_tokens > 0 for r in all_results)
    costs = [r.cost_per_million_tokens for r in all_results] if has_cost else []
    
    # Create figure with subplots - use 4x2 if cost data available, otherwise 3x2
    num_rows = 4 if has_cost else 3
    fig, axes = plt.subplots(num_rows, 2, figsize=(32, 12 if not has_cost else 16))
    fig.suptitle('QPS Benchmark Results', fontsize=16, fontweight='bold')
    
    # Plot 1: QPS (Target vs Actual) and Success Rate
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(qps_targets, actual_qps, 'o-', color='#2E86AB', linewidth=2, 
                     markersize=8, label='Actual QPS')
    ax1.plot(qps_targets, qps_targets, '--', color='#666', linewidth=1, alpha=0.5, label='Target QPS')
    
    line2 = ax1_twin.plot(qps_targets, success_rates, 's-', color='#A23B72', 
                          linewidth=2, markersize=8, label='Success Rate')
    
    ax1.set_xlabel('Target QPS', fontsize=12)
    ax1.set_ylabel('Actual QPS', fontsize=12, color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin.set_ylabel('Success Rate (%)', fontsize=12, color='#A23B72')
    ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
    ax1_twin.set_ylim([0, 105])
    ax1_twin.axhline(y=95, color='#A23B72', linestyle=':', alpha=0.5, linewidth=1)
    ax1_twin.text(qps_targets[0], 95, ' 95% threshold', 
                  verticalalignment='bottom', fontsize=9, color='#A23B72')
    
    # Combine legends from both axes and place at top center
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), framealpha=0.95, edgecolor='black', ncol=3)
    
    ax1.set_title('Throughput and Success Rate', fontsize=13, fontweight='bold')
    
    # Plot 2: TTFT Percentiles
    ax2 = axes[0, 1]
    ax2.plot(qps_targets, ttft_p50, 'o-', color='#06A77D', linewidth=2, 
             markersize=8, label='P50 (median)')
    ax2.plot(qps_targets, ttft_p90, 'd-', color='#2E86AB', linewidth=2, 
             markersize=8, label='P90')
    ax2.plot(qps_targets, ttft_p95, 's-', color='#F18F01', linewidth=2, 
             markersize=8, label='P95')
    ax2.plot(qps_targets, ttft_p99, '^-', color='#C73E1D', linewidth=2, 
             markersize=8, label='P99')
    
    ax2.set_xlabel('Target QPS', fontsize=12)
    ax2.set_ylabel('Time to First Token (seconds)', fontsize=12)
    ax2.set_title('Time to First Token (TTFT)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), framealpha=0.95, edgecolor='black', ncol=4)
    ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax2.text(qps_targets[0], 1.0, ' 1s threshold', 
             verticalalignment='bottom', fontsize=9, color='red')
    
    # Plot 3: End-to-End Latency
    ax3 = axes[1, 0]
    ax3.plot(qps_targets, e2e_p50, 'o-', color='#06A77D', linewidth=2, 
             markersize=8, label='P50 (median)')
    ax3.plot(qps_targets, e2e_p90, 'd-', color='#2E86AB', linewidth=2, 
             markersize=8, label='P90')
    ax3.plot(qps_targets, e2e_p95, 's-', color='#F18F01', linewidth=2, 
             markersize=8, label='P95')
    ax3.plot(qps_targets, e2e_p99, '^-', color='#C73E1D', linewidth=2, 
             markersize=8, label='P99')
    
    ax3.set_xlabel('Target QPS', fontsize=12)
    ax3.set_ylabel('End-to-End Latency (seconds)', fontsize=12)
    ax3.set_title('End-to-End Request Latency', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), framealpha=0.95, edgecolor='black', ncol=4)
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find saturation point
    saturation_idx = len(all_results) - 1
    for i, result in enumerate(all_results):
        if result.success_rate < 95:
            saturation_idx = i
            break
    
    # Create summary text
    summary_text = "Summary & Recommendations\n" + "="*50 + "\n\n"
    
    # Best performance
    best_idx = max(0, saturation_idx - 1) if saturation_idx > 0 else 0
    summary_text += f"Best Performance (before saturation):\n"
    summary_text += f"  Target QPS: {all_results[best_idx].qps_target:.1f}\n"
    summary_text += f"  Actual QPS: {all_results[best_idx].actual_qps:.2f}\n"
    summary_text += f"  Success Rate: {all_results[best_idx].success_rate:.1f}%\n"
    summary_text += f"  TTFT P50: {all_results[best_idx].ttft_p50:.3f}s\n"
    summary_text += f"  E2E P50: {all_results[best_idx].e2e_latency_p50:.3f}s\n\n"
    
    # Saturation point
    if saturation_idx < len(all_results):
        summary_text += f"Saturation Point:\n"
        summary_text += f"  QPS: {all_results[saturation_idx].qps_target:.1f}\n"
        summary_text += f"  Success Rate: {all_results[saturation_idx].success_rate:.1f}%\n"
        if all_results[saturation_idx].failed_requests > 0:
            summary_text += f"  Failed Requests: {all_results[saturation_idx].failed_requests}\n"
        summary_text += "\n"
    
    # Recommendations
    good_results = [r for r in all_results if r.success_rate >= 95]
    if good_results:
        max_good_qps = max(r.qps_target for r in good_results)
        summary_text += f"Recommended Production QPS:\n"
        summary_text += f"  Max QPS: {max_good_qps:.1f}\n"
        summary_text += f"  Conservative (80%): {max_good_qps * 0.8:.1f}\n"
        summary_text += f"  Aggressive (90%): {max_good_qps * 0.9:.1f}\n\n"
    
    summary_text += "Guidance:\n"
    summary_text += "  • Use conservative limit for production\n"
    summary_text += "  • Monitor success rate and latency\n"
    summary_text += "  • Scale horizontally if need more capacity\n"
    
    # Add error summary if any
    has_errors = any(r.failed_requests > 0 for r in all_results)
    if has_errors:
        summary_text += "\nErrors Detected:\n"
        for result in all_results:
            if result.failed_requests > 0:
                summary_text += f"  QPS {result.qps_target:.1f}: {result.failed_requests} failures\n"
                if result.error_counts:
                    top_error = max(result.error_counts.items(), key=lambda x: x[1])
                    summary_text += f"    Top error: {top_error[0]} ({top_error[1]}x)\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 5: Tokens/sec - always show this graph
    ax5 = axes[2, 0]
    ax5_twin = ax5.twinx()
    
    # Per-request percentiles (left y-axis)
    line1 = ax5.plot(qps_targets, tps_p50, 'o-', color='#06A77D', linewidth=2, 
                     markersize=8, label='P50 (per-req)')
    line2 = ax5.plot(qps_targets, tps_p90, 'd-', color='#2E86AB', linewidth=2, 
                     markersize=8, label='P90 (per-req)')
    line3 = ax5.plot(qps_targets, tps_p95, 's-', color='#F18F01', linewidth=2, 
                     markersize=8, label='P95 (per-req)')
    line4 = ax5.plot(qps_targets, tps_p99, '^-', color='#C73E1D', linewidth=2, 
                     markersize=8, label='P99 (per-req)')
    
    # Aggregate sustained throughput (right y-axis)
    line5 = ax5_twin.plot(qps_targets, sustained_throughput, 'D-', color='#8B008B', 
                          linewidth=3, markersize=10, label='Aggregate (all reqs)', 
                          markeredgewidth=2, markeredgecolor='white')
    
    ax5.set_xlabel('Target QPS', fontsize=12)
    ax5.set_ylabel('Per-Request Tokens/sec', fontsize=12, color='#06A77D')
    ax5.tick_params(axis='y', labelcolor='#06A77D')
    ax5.grid(True, alpha=0.3)
    
    # Add padding to top of per-request axis
    if tps_p99:
        max_per_req = max(tps_p99)
        ax5.set_ylim([0, max_per_req * 1.25])
    
    ax5_twin.set_ylabel('Aggregate Output Tokens/sec', fontsize=12, color='#8B008B', fontweight='bold')
    ax5_twin.tick_params(axis='y', labelcolor='#8B008B')
    
    # Add padding to top of aggregate axis
    if sustained_throughput:
        max_aggregate = max(sustained_throughput)
        ax5_twin.set_ylim([0, max_aggregate * 1.25])
    
    # Combine legends from both axes
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.03), framealpha=0.95, edgecolor='black', ncol=3)
    
    ax5.set_title('Generation Speed: Per-Request vs Aggregate', fontsize=13, fontweight='bold')
    
    # Plot 6: Cost per Million Tokens (if available)
    if has_cost:
        ax6 = axes[2, 1]
        
        # Plot cost line
        ax6.plot(qps_targets, costs, 'o-', color='#8B4789', linewidth=2, 
                 markersize=8, label='Cost per 1M tokens')
        
        # Add markers for success rate context
        for i, (qps, cost, success) in enumerate(zip(qps_targets, costs, success_rates)):
            if success < 95:
                ax6.plot(qps, cost, 'rx', markersize=12, markeredgewidth=2)
                ax6.annotate(f'{success:.0f}%', (qps, cost), 
                           textcoords="offset points", xytext=(0,10),
                           ha='center', fontsize=8, color='red')
        
        ax6.set_xlabel('Target QPS', fontsize=12)
        ax6.set_ylabel('Cost per Million Tokens ($)', fontsize=12)
        ax6.set_title('Cost per Million Tokens (Input + Output)\n(Based on Total Aggregate Throughput)', 
                     fontsize=13, fontweight='bold')
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), framealpha=0.95, edgecolor='black')
        ax6.grid(True, alpha=0.3)
        
        # Add explanatory text
        ax6.text(0.02, 0.98, '✗ = Success rate < 95% (need more GPUs)', 
                transform=ax6.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Hide empty subplots in 4x2 grid
        axes[3, 0].axis('off')
        axes[3, 1].axis('off')
    else:
        # Hide the 6th subplot in 3x2 grid
        ax6 = axes[2, 1]
        ax6.axis('off')
    
    # Adjust layout to make room for legends above plots
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="QPS-based benchmark for LLM endpoint - Find when your endpoint saturates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with single QPS level
  %(prog)s --endpoint https://api.example.com/v1/chat/completions \\
           --api-key sk-xxxx \\
           --model meta-llama/Llama-3.1-8B-Instruct \\
           --dataset conversations.json \\
           --qps 10 \\
           --duration 60

  # Test multiple QPS levels to find saturation point
  %(prog)s --endpoint https://api.example.com/v1/chat/completions \\
           --api-key sk-xxxx \\
           --model meta-llama/Llama-3.1-8B-Instruct \\
           --dataset conversations.json \\
           --qps 1,5,10,20,50 \\
           --duration 60

  # Generate visualization with results
  %(prog)s --endpoint https://api.example.com/v1/chat/completions \\
           --api-key sk-xxxx \\
           --model meta-llama/Llama-3.1-8B-Instruct \\
           --dataset conversations.json \\
           --qps 1,5,10,20,50 \\
           --duration 60 \\
           --plot

  # Include tokens/sec metrics in output
  %(prog)s --endpoint https://api.example.com/v1/chat/completions \\
           --api-key sk-xxxx \\
           --model meta-llama/Llama-3.1-8B-Instruct \\
           --dataset conversations.json \\
           --qps 1,5,10 \\
           --show-tokens

  # Show only P50 and P90 percentiles
  %(prog)s --endpoint https://api.example.com/v1/chat/completions \\
           --api-key sk-xxxx \\
           --model meta-llama/Llama-3.1-8B-Instruct \\
           --dataset conversations.json \\
           --qps 1,5,10 \\
           --duration 60 \\
           --percentiles 50,90

  # Compact output with tokens/sec
  %(prog)s --endpoint https://api.example.com/v1/chat/completions \\
           --api-key sk-xxxx \\
           --model meta-llama/Llama-3.1-8B-Instruct \\
           --dataset conversations.json \\
           --qps 1,5,10,20 \\
           --percentiles 50 \\
           --show-tokens

Requirements:
  pip install aiohttp tqdm
  pip install matplotlib  # Optional, for --plot

What to Look For:
  - Success rate drops below 95%% = Saturation point
  - HTTP 503 errors = Endpoint overloaded
  - TTFT increases significantly = GPU queuing requests
  
Recommended: Start with low QPS and increase to find your limit.
        """
    )
    
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="API endpoint URL",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for authentication",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file containing conversation samples",
    )
    
    parser.add_argument(
        "--qps",
        type=str,
        default="1,5,10,20",
        help="Comma-separated list of QPS levels to test (default: 1,5,10,20)",
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration to run each QPS test in seconds (default: 60)",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per request (default: 512)",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_qps_results.json",
        help="Output file for results (default: benchmark_qps_results.json)",
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization PNG (requires matplotlib: pip install matplotlib)",
    )
    
    parser.add_argument(
        "--plot-output",
        type=str,
        default="benchmark_qps.png",
        help="Output file for visualization (default: benchmark_qps.png)",
    )
    
    parser.add_argument(
        "--percentiles",
        type=str,
        default="50,95,99",
        help="Comma-separated percentiles to display (default: 50,95,99). Available: 50,90,95,99",
    )
    
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show tokens/sec metrics in output table and visualization",
    )
    
    parser.add_argument(
        "--gpu-rate",
        type=float,
        help="GPU cost per hour (e.g., 2.30 for $2.30/hour). Required for cost calculation.",
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs used. Required for cost calculation.",
    )
    
    args = parser.parse_args()
    
    # Create timestamped results directory
    results_dir = create_results_directory()
    
    # Update output paths to use results directory
    output_file = os.path.join(results_dir, os.path.basename(args.output))
    plot_output_file = os.path.join(results_dir, os.path.basename(args.plot_output))
    
    # Parse QPS levels
    qps_levels = [float(q.strip()) for q in args.qps.split(',')]
    
    # Parse percentiles
    percentiles = [int(p.strip()) for p in args.percentiles.split(',')]
    # Validate percentiles
    valid_percentiles = [50, 90, 95, 99]
    for p in percentiles:
        if p not in valid_percentiles:
            print(f"❌ Error: Invalid percentile {p}. Must be one of: {valid_percentiles}")
            return
    
    print("="*80)
    print("QPS BENCHMARK")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"QPS levels: {qps_levels}")
    print(f"Duration per test: {args.duration} seconds")
    print("="*80)
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    if not dataset:
        print("❌ Error: No data loaded from dataset")
        return
    
    # Run QPS tests
    all_results = []
    
    for qps in qps_levels:
        result = await run_qps_test(
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            dataset=dataset,
            qps_target=qps,
            duration=args.duration,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_rate=args.gpu_rate,
            num_gpus=args.num_gpus,
        )
        all_results.append(result)
        
        # Show immediate feedback
        print(f"\n✅ Completed: {result.successful_requests}/{result.total_requests} successful ({result.success_rate:.1f}%)")
        
        # Stop if we hit significant failures
        if result.success_rate < 50:
            print(f"\n⚠️  WARNING: Success rate dropped below 50% at QPS {qps}")
            print("Stopping tests - endpoint is overloaded.")
            break
    
    # Save metadata about this benchmark run
    metadata_file = os.path.join(results_dir, "metadata.json")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": args.endpoint,
        "model": args.model,
        "dataset": args.dataset,
        "qps_levels": qps_levels,
        "duration": args.duration,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "percentiles": percentiles,
        "show_tokens": args.show_tokens,
        "token_counting": "API-reported only (no estimates)",
        "cost_calculation": "Total tokens (input + output) from API usage data",
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print and save results
    show_cost = args.gpu_rate is not None and args.num_gpus is not None
    print_results_table(all_results, percentiles, show_cost)
    save_results(all_results, output_file)
    
    # Generate visualization if requested
    if args.plot:
        plot_results(all_results, plot_output_file)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in all_results:
        status = "✅ PASS" if result.success_rate >= 95 else "⚠️  DEGRADED" if result.success_rate >= 80 else "❌ FAIL"
        print(f"QPS {result.qps_target:5.1f}: {status} - {result.success_rate:.1f}% success, {result.actual_qps:.2f} actual QPS")
    
    # Find saturation point
    good_results = [r for r in all_results if r.success_rate >= 95]
    if good_results:
        max_good_qps = max(r.qps_target for r in good_results)
        print(f"\n🎯 Recommended max QPS: {max_good_qps:.1f} (maintains >95% success rate)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())

