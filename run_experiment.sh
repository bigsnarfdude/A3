#!/bin/bash
# A3 Experiment Runner — Testing unsloth + claude -p patches
# No OpenRouter needed. Serves base model locally with vLLM for step 2,
# then kills it and unsloth takes over for step 4.
#
# Usage:
#   bash run_experiment.sh                  # 10 iterations
#   bash run_experiment.sh 5                # quick smoke test
#
# Prerequisites:
#   - bash setup_a100.sh completed
#   - ANTHROPIC_API_KEY set
#   - claude CLI installed

set -euo pipefail

ITERATIONS=${1:-10}
CONFIG="configs/sycophancy-llama-nigel.json"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
VLLM_PORT=8000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment_logs"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "============================================" | tee "$LOG_FILE"
echo "A3 EXPERIMENT — Unsloth + claude -p path" | tee -a "$LOG_FILE"
echo "Config: $CONFIG" | tee -a "$LOG_FILE"
echo "Iterations: $ITERATIONS" | tee -a "$LOG_FILE"
echo "No OpenRouter — all local" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

if ! command -v claude &> /dev/null; then
    echo "ERROR: claude CLI not found. Run setup_a100.sh first"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Helper: start vLLM and wait for health
start_vllm() {
    echo "Starting vLLM server for $BASE_MODEL ..." | tee -a "$LOG_FILE"
    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --trust-remote-code &
    VLLM_PID=$!

    echo "Waiting for vLLM (PID $VLLM_PID) ..." | tee -a "$LOG_FILE"
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM ready (${i}s)" | tee -a "$LOG_FILE"
            return 0
        fi
        # Check if process died
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM process died" | tee -a "$LOG_FILE"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM timeout" | tee -a "$LOG_FILE"
    return 1
}

kill_vllm() {
    echo "Killing vLLM ..." | tee -a "$LOG_FILE"
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    # Force kill stragglers
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
    echo "vLLM stopped. GPU free." | tee -a "$LOG_FILE"
}

# Cleanup on exit
trap kill_vllm EXIT

# ============================================
# Step 1: Data Generation (API only, no GPU)
# ============================================
echo "STEP 1: Data Generation — $(date)" | tee -a "$LOG_FILE"

python scripts/step1_data_generation.py \
    --config-file "$CONFIG" \
    2>&1 | tee -a "$LOG_FILE"

echo "Step 1 done: $(date)" | tee -a "$LOG_FILE"

# ============================================
# Step 2: Evaluation & Splitting
# Start vLLM to serve base model, eval, then kill it
# ============================================
echo "STEP 2: Evaluation — $(date)" | tee -a "$LOG_FILE"

start_vllm

python scripts/step2_evaluation.py \
    --config-file "$CONFIG" \
    2>&1 | tee -a "$LOG_FILE"

kill_vllm

echo "Step 2 done: $(date)" | tee -a "$LOG_FILE"

# ============================================
# Step 3: Generate Expected Behaviors (API only)
# ============================================
echo "STEP 3: Expected Behaviors — $(date)" | tee -a "$LOG_FILE"

python scripts/step3_generate_expected_behaviors.py \
    --config-file "$CONFIG" \
    2>&1 | tee -a "$LOG_FILE"

echo "Step 3 done: $(date)" | tee -a "$LOG_FILE"

# ============================================
# Step 4: Iterative SFT (unsloth, single GPU)
# No vLLM — unsloth does train + inference in one process
# ============================================
echo "STEP 4: SFT Training ($ITERATIONS iterations) — $(date)" | tee -a "$LOG_FILE"

python scripts/step4_sft_agent.py \
    --config-file "$CONFIG" \
    --num-iterations "$ITERATIONS" \
    2>&1 | tee -a "$LOG_FILE"

echo "Step 4 done: $(date)" | tee -a "$LOG_FILE"

# ============================================
# Done
# ============================================
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "EXPERIMENT COMPLETE — $(date)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
ls -d sft_results_* evaluation_results_* 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Pull: bash pull_results.sh user@local:~/Desktop/a3-results/"
