#!/bin/bash
# A3 Setup Script — Single A100 40GB (testing unsloth + claude -p patches)
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash setup_a100.sh
#
# Then run:
#   bash run_experiment.sh

set -euo pipefail

echo "============================================"
echo "A3 SETUP — Single A100 40GB (unsloth path)"
echo "============================================"

# Check claude CLI (all API calls go through claude -p, no API key needed)
if ! command -v claude &> /dev/null; then
    echo "Installing Claude CLI..."
    npm install -g @anthropic-ai/claude-code
fi

echo "Claude CLI: $(which claude)"

# Check GPU
echo ""
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install deps
echo "Installing dependencies..."
pip install --upgrade pip
pip install requests tiktoken
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets
pip install trl bitsandbytes
pip install "unsloth[cu121-torch250]"
pip install vllm        # for baseline eval in step 2
pip install lm-eval     # for MMLU-Pro and GPQA benchmarks

echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next: bash run_experiment.sh"
echo ""
