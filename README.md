# LLM Endpoint Benchmark Tool

A QPS (Queries Per Second) based benchmark tool for measuring LLM endpoint performance and finding saturation points.

## Purpose

This benchmark helps you:
- **Find your endpoint's capacity** - Discover the maximum QPS your endpoint can handle
- **Identify saturation points** - See when your endpoint starts returning 503 errors or timing out  
- **Measure latency** - Track Time-to-First-Token (TTFT), End-to-End (E2E) latency, and Tokens/sec at different load levels
- **Validate performance** - Test with real conversation data to simulate production workloads

## Quick Start

### 1. Install Dependencies

```bash
pip install aiohttp tqdm matplotlib
```

### 2. Create Your Dataset

Create a JSON file with conversation samples (see Dataset section below for examples):

```bash
# Example: conversations.json
[
  {
    "prompt_content": "You are a helpful assistant.",
    "cut_chat_history": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi! How can I help?"},
      {"role": "user", "content": "What's the weather like?"}
    ]
  }
]
```

### 3. Run Benchmark

```bash
python benchmark.py \
    --endpoint "https://your-endpoint.com/v1/chat/completions" \
    --api-key "your-api-key" \
    --model "your-model-name" \
    --dataset conversations.json \
    --qps 1,5,10,20 \
    --duration 60
```

### 4. View Results

The benchmark automatically creates a timestamped results directory for each run:

```
results/
├── 2024-11-14_10-30-45/
│   ├── metadata.json                    # Run configuration and metadata
│   ├── benchmark_qps_results.json       # Detailed benchmark results
│   └── benchmark_qps.png                # Visualization (if --plot used)
├── 2024-11-14_11-15-22/
│   ├── metadata.json
│   ├── benchmark_qps_results.json
│   └── benchmark_qps.png
```

**Each results directory contains:**
- **Results table** printed to console
- **metadata.json** - Run configuration (timestamp, model, dataset, QPS levels)
- **benchmark_qps_results.json** - Complete benchmark data with all metrics
- **benchmark_qps.png** - Visualization graphs (when `--plot` is used)

This structure keeps your benchmark history organized and prevents overwriting previous results.

## Dataset Format

The benchmark expects a JSON file with an array of conversation samples. Each sample represents a complete chat conversation that will be sent to your LLM endpoint.

### Example Dataset with Different Conversation Types

```json
[
  {
    "prompt_content": "You are a helpful assistant. Answer questions clearly and concisely.",
    "cut_chat_history": [
      {
        "role": "user",
        "content": "What is Python?"
      }
    ]
  },
  {
    "prompt_content": "You are a coding tutor. Help users learn programming.",
    "cut_chat_history": [
      {
        "role": "user",
        "content": "How do I write a for loop?"
      },
      {
        "role": "assistant",
        "content": "A for loop in Python looks like this: for i in range(10): print(i)"
      },
      {
        "role": "user",
        "content": "Can you explain what range does?"
      }
    ]
  },
  {
    "prompt_content": "You are a customer service agent. Be professional and helpful.",
    "cut_chat_history": [
      {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      {
        "role": "user",
        "content": "I need help with my order"
      },
      {
        "role": "assistant",
        "content": "I'd be happy to help. Can you provide your order number?"
      },
      {
        "role": "user",
        "content": "It's ORD-12345"
      }
    ]
  }
]
```

### Understanding Different Conversation Types

#### 1. **Simple Question** (Single Turn)
```json
{
  "prompt_content": "You are a helpful assistant.",
  "cut_chat_history": [
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```
**Sent to LLM as:**
```
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is 2+2?"}
]
```
**Use case:** Tests baseline response time for simple queries.

#### 2. **Multi-Turn Dialogue** (Conversation with Context)
```json
{
  "prompt_content": "You are a friendly chatbot.",
  "cut_chat_history": [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How are you today?"},
    {"role": "user", "content": "I'm good. Can you help me with something?"},
    {"role": "assistant", "content": "Of course! What do you need help with?"},
    {"role": "user", "content": "I need recipe suggestions"}
  ]
}
```
**Sent to LLM as:**
```
[
  {"role": "system", "content": "You are a friendly chatbot."},
  {"role": "user", "content": "Hi!"},
  {"role": "assistant", "content": "Hello! How are you today?"},
  {"role": "user", "content": "I'm good. Can you help me with something?"},
  {"role": "assistant", "content": "Of course! What do you need help with?"},
  {"role": "user", "content": "I need recipe suggestions"}
]
```
**Use case:** Tests how your endpoint handles longer context windows. As conversations progress, input token count increases, which affects latency and throughput.

#### 3. **Complex Multi-Turn with Long Context**
```json
{
  "prompt_content": "You are an expert software architect. Provide detailed technical advice.",
  "cut_chat_history": [
    {"role": "user", "content": "I'm building a microservices architecture..."},
    {"role": "assistant", "content": "Great! Let me help you design that..."},
    {"role": "user", "content": "What about database sharding?"},
    {"role": "assistant", "content": "Database sharding is an excellent strategy..."},
    {"role": "user", "content": "How do I handle distributed transactions?"}
  ]
}
```
**Use case:** Tests performance with substantial context, similar to real production scenarios where users have extended conversations.

### How Conversations Are Sent to Your LLM

For each request in the benchmark:

1. **Load a sample** from your dataset (cycles through all samples)
2. **Convert to messages format**:
   - System message from `prompt_content`
   - All turns from `cut_chat_history` (excluding system role turns)
3. **Send to your endpoint** via POST request:

```python
POST /v1/chat/completions
{
  "model": "your-model",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": true
}
```

4. **Measure response**:
   - **TTFT**: Time until first token arrives (streaming)
   - **E2E Latency**: Total time from request to completion
   - **Tokens/sec**: Output tokens ÷ total time
   - **Success/Failure**: HTTP status codes, errors

### Why Different Conversation Types Matter

| Conversation Type | Input Tokens | Impact on Performance |
|-------------------|--------------|----------------------|
| **Single turn** | Low (~50-200) | Fastest response, tests baseline speed |
| **Multi-turn (3-5 turns)** | Medium (~200-500) | Realistic for most conversations |
| **Long context (10+ turns)** | High (~500-2000) | Tests context window handling, slower |

**Key insight**: Longer conversations = more input tokens = slower TTFT and lower throughput. Your dataset should reflect your real production distribution of conversation lengths.

### Required Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `prompt_content` | string | System prompt prepended to conversation | Yes |
| `cut_chat_history` | array | Array of conversation turns | Yes |

Each turn in `cut_chat_history`:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `role` | string | "user", "assistant", or "system" | Yes |
| `content` | string | The message content | Yes |

**Note:** System messages in `cut_chat_history` are skipped (only the initial `prompt_content` becomes the system message).

### Optional Fields

Additional fields like `isAssistantMessage`, `turnXp`, `animation`, etc. are ignored by the benchmark.

## Creating Your Dataset

### Option 1: Sample Real Production Data

Export real conversations from your production system:
- Preserve conversation flow (user/assistant turns)
- Include diverse conversation lengths (short, medium, long)
- Anonymize sensitive data
- Save as JSON in the required format

### Option 2: Generate Synthetic Data

Create realistic synthetic conversations for testing:

```python
import json

conversations = []

# Add simple queries
for topic in ["weather", "time", "math", "facts"]:
    conversations.append({
        "prompt_content": "You are a helpful assistant.",
        "cut_chat_history": [
            {"role": "user", "content": f"Tell me about {topic}"}
        ]
    })

# Add multi-turn conversations
conversations.append({
    "prompt_content": "You are a coding tutor.",
    "cut_chat_history": [
        {"role": "user", "content": "How do I learn Python?"},
        {"role": "assistant", "content": "Start with basics like variables and loops."},
        {"role": "user", "content": "What are variables?"}
    ]
})

# Save to file
with open('conversations.json', 'w') as f:
    json.dump(conversations, f, indent=2)
```

### Option 3: Minimal Test Dataset

For quick testing, create a minimal dataset:

```json
[
  {
    "prompt_content": "You are a helpful AI assistant.",
    "cut_chat_history": [
      {"role": "user", "content": "Hello"}
    ]
  },
  {
    "prompt_content": "You are a helpful AI assistant.",
    "cut_chat_history": [
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I help?"},
      {"role": "user", "content": "What's the weather?"}
    ]
  },
  {
    "prompt_content": "You are a helpful AI assistant.",
    "cut_chat_history": [
      {"role": "user", "content": "Tell me about Python"},
      {"role": "assistant", "content": "Python is a programming language."},
      {"role": "user", "content": "What are its features?"},
      {"role": "assistant", "content": "Python is easy to learn and versatile."},
      {"role": "user", "content": "How do I get started?"}
    ]
  }
]
```

## Usage

### Basic Benchmark

```bash
python benchmark.py \
    --endpoint "https://api.example.com/v1/chat/completions" \
    --api-key "sk-xxxx" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset conversations.json \
    --qps 1,5,10,20 \
    --duration 60
```

### With Visualization

```bash
python benchmark.py \
    --endpoint "https://api.example.com/v1/chat/completions" \
    --api-key "sk-xxxx" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset conversations.json \
    --qps 1,5,10,20 \
    --duration 60 \
    --plot
```

### Compact Output (P50 only)

```bash
python benchmark.py \
    --endpoint "https://api.example.com/v1/chat/completions" \
    --api-key "sk-xxxx" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset conversations.json \
    --qps 1,5,10,20 \
    --percentiles 50
```

## Command Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--endpoint` | API endpoint URL | - | Yes |
| `--api-key` | Authentication key | - | Yes |
| `--model` | Model name | - | Yes |
| `--dataset` | Path to dataset JSON | - | Yes |
| `--qps` | QPS levels to test | `1,5,10,20` | No |
| `--duration` | Seconds per test | `60` | No |
| `--percentiles` | Which percentiles to show | `50,95,99` | No |
| `--show-tokens` | (Deprecated - tokens/sec always shown) | - | No |
| `--max-tokens` | Max tokens per request | `512` | No |
| `--temperature` | Generation temperature | `0.7` | No |
| `--plot` | Generate visualization | Off | No |
| `--output` | JSON results file | `benchmark_qps_results.json` | No |
| `--plot-output` | PNG visualization file | `benchmark_qps.png` | No |
| `--gpu-rate` | GPU cost per hour (e.g., 2.30) | None | No |
| `--num-gpus` | Number of GPUs used | None | No |

**Note:** Output files are automatically placed in timestamped directories under `results/`. You don't need to worry about overwriting previous results.

### Cost Calculation

To calculate cost per million tokens, provide both `--gpu-rate` and `--num-gpus`:

```bash
python benchmark.py \
    --endpoint "https://api.example.com/v1/chat/completions" \
    --api-key "sk-xxxx" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset conversations.json \
    --qps 10,20,30 \
    --duration 60 \
    --gpu-rate 2.30 \
    --num-gpus 2
```

**Cost Calculation Formula (Simplified and Accurate):**

The benchmark calculates cost based on **actual API-reported tokens only** (no estimates):

1. **Total tokens in run** = sum of input + output tokens from API 'usage' field (successful requests only)
2. **GPU cost for run** = num_gpus × rate_per_hour × (actual_duration / 3600)
3. **Cost per 1M tokens** = (GPU_cost_for_run ÷ total_tokens_in_run) × 1,000,000

**Formula in code:**
```
cost_per_1m = (num_gpus × GPU_rate × duration_hours) × (1,000,000 / total_tokens)
```

**Why this approach?**
- Simple and accurate: scales the actual cost of your test run to 1M tokens
- No assumptions about GPU throughput rates
- Works regardless of input/output ratio
- Direct proportional scaling

**Important Notes:**
- Only uses token counts reported by the API in the `usage` field
- No estimates or approximations
- If requests lack token data, they're excluded and a warning is shown
- Works with any input/output ratio (e.g., 130:1 or 1:10)

This gives you the **real cost per million total tokens** based on your actual workload characteristics.

**Example:**
- Run: 30 seconds at 40 QPS
- GPUs: 1 × $2.30/hour
- Tokens processed:
  - Input: 3,659,418 tokens
  - Output: 28,146 tokens
  - Total: 3,687,564 tokens
  - Ratio: 130:1 (input:output)

**Calculation:**
```
1. GPU cost for 30-second run:
   $2.30/hr × 1 GPU × (30s / 3600s) = $0.0192

2. Cost per million tokens:
   ($0.0192 / 3,687,564 tokens) × 1,000,000 = $0.0052 per million tokens
```

**Why Count Both Input and Output?**
- GPU spends time on BOTH:
  - **Prefill** (input): Processes all input tokens in parallel (fast, but uses GPU cycles)
  - **Decode** (output): Generates output tokens one-by-one (slower, but less total tokens)
- With a 130:1 input/output ratio, most GPU time is prefill
- Cost reflects actual GPU runtime for your specific workload
- This matches how cloud providers charge (total tokens, not just output)

**Why Does Cost Per Token Decrease at Higher QPS?**

This is counter-intuitive but correct! Here's why:

| QPS Level | GPU Utilization | Total Tok/Sec | Cost per 1M Tokens | Why |
|-----------|-----------------|---------------|-------------------|-----|
| **5 QPS**  | ~10% (mostly idle) | 2,000 | $4.00 | GPU waits between requests |
| **20 QPS** | ~50% (moderate) | 20,000 | $0.40 | Some request overlap |
| **45 QPS** | ~90% (busy) | 70,000 | $0.01 | Continuous batched processing |

**Key Insight:** You pay for GPU time whether it's busy or idle!

- At low QPS: GPU processes request → idles 95% of time → next request
- At high QPS: Requests overlap → GPU batches them → processes tokens continuously

** Important Notes:**
- Cost is calculated per million **TOTAL** tokens (input + output combined)
- Both prefill (input) and decode (output) consume GPU time and contribute to cost
- If success rate < 95%, the cost assumes you can sustain the measured throughput, but errors indicate you'd need MORE GPUs to reliably handle that rate
- Higher QPS with errors means the single-GPU cost looks cheap, but you'd actually need more GPUs for production
- Compare costs at the highest QPS with 99%+ success rate for most accurate production estimates

**Cost is displayed in:**
- Console table ($/M Tokens column)
- JSON results (cost_analysis section)
- Visualization graph (Cost per Million Tokens plot)

### Finding Your Optimal Configuration

To accurately determine cost per million tokens for production:

**Step 1: Find Saturation Point**
```bash
# Test increasing QPS levels to find where errors start
python benchmark.py \
    --endpoint "https://your-endpoint.com" \
    --api-key "your-key" \
    --model "your-model" \
    --dataset conversations.json \
    --qps 5,10,15,20,25,30 \
    --duration 60 \
    --gpu-rate 2.30 \
    --num-gpus 1
```

**Step 2: Identify Maximum Sustainable QPS**

Look for the **highest QPS with ≥99% success rate**. Example output:

```
QPS 5.0:  PASS - 100% success, 2,000 total tok/s (~1% GPU util), $4.00/M   ← Too low!
QPS 10.0: PASS - 100% success, 12,000 total tok/s (~15% GPU util), $0.67/M
QPS 30.0: PASS - 99.5% success, 48,000 total tok/s (~60% GPU util), $0.17/M  ← Good!
QPS 45.0: PASS - 99.8% success, 70,500 total tok/s (~88% GPU util), $0.01/M  ← Optimal!
QPS 50.0:  WARN - 92.1% success, 65,000 total tok/s, $0.013/M tokens  ← Saturated!
QPS 60.0: FAIL - 45.2% success, 48,000 total tok/s, $0.018/M tokens  ← Overloaded!
```

**Use QPS 45.0** for production cost estimates ($0.01/M total tokens with 1 GPU)

**Why not use QPS 5.0 even though it's 100% success?**
- The GPU is 99% idle, waiting for requests
- You're paying $2.30/hour for a GPU that's barely working
- Cost per token is artificially high due to underutilization
- At 45 QPS, you get ~400x cheaper cost per token with the same GPU!

**Step 3: Scale Calculation**

If you need higher throughput than one GPU can sustain:

| GPUs | Max Sustainable QPS | Total Tokens/sec | Cost/M Total Tokens | Total Throughput |
|------|---------------------|------------------|---------------------|------------------|
| 1    | 45 QPS              | 70,500           | $0.01               | 254M tok/hour   |
| 2    | 90 QPS              | 141,000          | $0.01*              | 508M tok/hour   |
| 4    | 180 QPS             | 282,000          | $0.01*              | 1,015M tok/hour |

*Assuming linear scaling (test to verify!)

**Step 4: Production Planning**

```
Your requirements: 100M tokens/day
Sustained throughput needed: 100M / 24 hours = 4.17M tokens/hour

From table above:
- 1 GPU handles 18.8M/hour → Sufficient! ✅
- Daily cost: (100M / 1M) × $0.16 = $16/day
```

## Working with Results

### Results Directory Structure

Each benchmark run creates a timestamped directory:

```
results/2024-11-14_10-30-45/
```

### Files in Results Directory

#### 1. metadata.json
Contains information about the benchmark run:

```json
{
  "timestamp": "2024-11-14T10:30:45.123456",
  "endpoint": "https://api.example.com/v1/chat/completions",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "conversations.json",
  "qps_levels": [1.0, 5.0, 10.0, 20.0],
  "duration": 60,
  "max_tokens": 512,
  "temperature": 0.7,
  "percentiles": [50, 95, 99],
  "show_tokens": true
}
```

Use this to quickly identify what was tested in each run.

#### 2. benchmark_qps_results.json
Complete benchmark data with all metrics (see JSON Output section below).

#### 3. benchmark_qps.png
High-resolution visualization with large, easy-to-read graphs showing:
- **QPS and Success Rate** - Throughput vs target with success rate overlay
- **TTFT Percentiles** - Time to first token (P50, P90, P95, P99)
- **E2E Latency Percentiles** - End-to-end request latency (P50, P90, P95, P99)
- **Generation Speed** - Dual y-axis showing:
  - **Per-Request Tokens/sec** (P50, P90, P95, P99) - User-facing speed
  - **Aggregate Throughput** (Purple line) - System capacity (total output tokens/sec)
- **Cost per Million Tokens** - Shows when `--gpu-rate` and `--num-gpus` are provided
- **Summary and Recommendations** - Optimal QPS, saturation point, guidance

The graphs are sized for clarity (32x12 or 32x16 inches, wide format) with 3x2 layout (standard) or 4x2 layout (with cost). All metrics (TTFT, E2E, Tokens/sec) are always displayed in graphs.

## Understanding Results

### Console Output

All metrics including tokens/sec are always displayed:

```
================================================================================
QPS BENCHMARK RESULTS
================================================================================
QPS Target   Actual QPS   Success%   TTFT P50(s)  TTFT P95(s)  TTFT P99(s)  E2E P50(s)   E2E P95(s)   E2E P99(s)   Tok/s P50    Tok/s P95    Tok/s P99   
1.0          1.00         100.0      0.120        0.150        0.180        2.500        2.800        3.000        85.3         92.1         98.5        
5.0          4.98         100.0      0.180        0.250        0.300        3.200        3.600        4.000        78.2         85.0         90.1        
10.0         9.95         98.5       0.250        0.400        0.500        4.100        4.800        5.200        68.5         75.3         80.2        
20.0         19.82        85.2       0.400        1.200        2.500        5.500        8.000        12.000       55.0         62.0         70.0        

QPS 20.0 - Errors (178 total):
  - HTTP 503: 145 occurrences
```

**With Cost Calculation** (when `--gpu-rate` and `--num-gpus` are provided):

```
QPS Target   Actual QPS   Success%   TTFT P50(s)  E2E P50(s)   Tok/s P50    $/M Tokens  
10.0         9.95         98.5       0.250        4.100        68.5         $0.0234     
20.0         19.82        85.2       0.400        5.500        55.0         $0.0445     
```

### Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **QPS Target** | Target queries per second | - |
| **Actual QPS** | Actual rate achieved | Close to target |
| **Success %** | Successful requests | >95% |
| **TTFT** | Time to First Token | <0.5s for real-time |
| **E2E Latency** | End-to-End request time | <5s for most models |
| **Tokens/sec** | Generation speed | >50 for good performance |
| **$/M Tokens** | Cost per million tokens | Depends on model/hardware |
| **P50** | Median (50th percentile) | Typical performance |
| **P90** | 90th percentile | Good performance |
| **P95** | 95th percentile | SLA target |
| **P99** | 99th percentile | Worst-case |

### Understanding Tokens/sec: Two Different Metrics

The benchmark tracks **two different tokens/sec metrics** that serve different purposes:

#### 1. Per-Request Tokens/sec (Graph Percentiles: P50, P90, P95, P99)

**What it measures**: How fast each individual request generates tokens.

- **Calculated as**: `output_tokens / request_total_time` (for each request)
- **Shown in graph**: Left y-axis with percentile distribution
- **Typical values**:
  - 7B models on H100: 80-150 tokens/sec per request
  - 70B models on H100: 40-80 tokens/sec per request
- **What affects it**:
  - Model size (larger = slower)
  - Input length (longer context = slower)
  - Queue wait time (if GPU is busy)
  - Request-specific batching effects

**Why it matters**: This is **user-facing latency** - each user sees tokens streaming at this speed. Lower per-request tokens/sec means users wait longer to see each word appear.

**Example**: If P50 = 100 tok/s, half your users see tokens appearing at ≥100 tokens/sec.

#### 2. Aggregate Sustained Throughput (Purple Line on Graph)

**What it measures**: Total system output token generation rate across ALL concurrent requests.

- **Calculated as**: `total_output_tokens / test_duration` (across all requests)
- **Shown in graph**: Right y-axis, bold purple line
- **Typical values**:
  - 1 H100 with 8B model: 5,000-15,000 aggregate tokens/sec
  - Scales with concurrent requests and GPU utilization
- **What affects it**:
  - Number of concurrent requests (QPS)
  - GPU utilization (batching efficiency)
  - Success rate (failures don't contribute)

**Why it matters**: This is **system capacity** - how many total tokens your GPU generates per second. Used for capacity planning and cost calculations.

**Example**: At 40 QPS with aggregate 10,000 tok/s, your GPU generates 10,000 output tokens every second across all concurrent requests.

#### Key Difference

```
Per-Request (100 tok/s):  Each user sees ~100 tokens/sec streaming
Aggregate (10,000 tok/s): System generates 10k tokens/sec total across 100 concurrent users

Both are correct! One measures user experience, the other measures system capacity.
```

### JSON Output

Results are saved with all metrics including tokens/sec and cost (if calculated):

```json
{
  "qps_target": 10.0,
  "actual_qps": 9.95,
  "success_rate": 98.5,
  "ttft": {
    "p50": 0.250,
    "p90": 0.350,
    "p95": 0.400,
    "p99": 0.500
  },
  "e2e_latency": {
    "p50": 4.100,
    "p90": 4.500,
    "p95": 4.800,
    "p99": 5.200
  },
  "tokens_per_sec": {
    "p50": 68.5,
    "p90": 72.1,
    "p95": 75.3,
    "p99": 80.2
  },
  "cost_analysis": {
    "total_input_tokens": 245000,
    "total_output_tokens": 155000,
    "total_tokens": 400000,
    "gpu_cost": 0.00935,
    "cost_per_million_tokens": 0.0234
  },
  "errors": {"HTTP 503": 9}
}
```

## Expected Performance

### Llama 3.1 8B on Single H100
- **Max QPS without 503s**: 15-30 QPS
- **Saturation point**: 30-50 QPS
- **TTFT at low load**: 0.1-0.3s
- **E2E latency**: 2-5s (depends on output length)
- **Tokens/sec**: 80-150 tokens/sec

### Llama 3.1 70B on Single H100
- **Max QPS without 503s**: 5-15 QPS
- **Saturation point**: 15-25 QPS
- **TTFT at low load**: 0.2-0.5s
- **E2E latency**: 5-10s (depends on output length)
- **Tokens/sec**: 40-80 tokens/sec

## Troubleshooting

### "Missing --dataset argument"
**Solution**: Dataset is required. Provide path to your conversations JSON file.

### "All requests fail"
**Causes**: Wrong endpoint URL, invalid API key, wrong model name
**Solution**: Verify credentials and test with single request: `--qps 0.1 --duration 10`

### "Tokens/sec is 0"
**Cause**: Model not generating tokens or streaming not working
**Solution**: Check endpoint supports streaming and returns tokens

### "50 QPS succeeds but seems wrong"
**Possible causes**: Responses cached, multiple GPUs, output truncated
**Solution**: Check actual token counts in results, verify endpoint configuration

## Tips

1. **Start with realistic data** - Use conversations that match your production workload
2. **Mix conversation lengths** - Include short, medium, and long conversations
3. **Test incrementally** - Start at QPS 1, gradually increase
4. **Watch for 503s** - First sign of capacity limit
5. **Monitor tokens/sec** - Generation speed is always displayed in both console and graphs
6. **Leave headroom** - Set production limit at 70-80% of max QPS
7. **Keep results organized** - Each run automatically creates a timestamped directory
8. **Compare runs** - Use metadata.json to quickly identify what was tested
