#!/bin/bash
# Pull all A3 experiment artifacts to local machine.
#
# Run FROM the A100 box:
#   bash pull_results.sh user@local:~/Desktop/a3-results/
#
# Or run FROM local machine:
#   scp -r user@a100-box:~/A3/sft_results_* ~/Desktop/a3-results/
#   scp -r user@a100-box:~/A3/sft_data_* ~/Desktop/a3-results/
#   scp -r user@a100-box:~/A3/evaluation_results_* ~/Desktop/a3-results/
#   scp -r user@a100-box:~/A3/experiment_logs/ ~/Desktop/a3-results/

set -euo pipefail

DEST=${1:?"Usage: bash pull_results.sh user@host:path/"}

echo "Pulling results to $DEST ..."

# Results and metrics
rsync -avz --progress sft_results_*/ "$DEST/sft_results/"
rsync -avz --progress evaluation_results_*/ "$DEST/evaluation_results/"
rsync -avz --progress experiment_logs/ "$DEST/experiment_logs/"

# Training data (for reproducibility)
rsync -avz --progress sft_data_*/ "$DEST/sft_data/"

# Config used
rsync -avz --progress configs/ "$DEST/configs/"

# LoRA adapters (small, useful)
find . -path "*/lora_adapters" -type d | while read d; do
    rsync -avz --progress "$d/" "$DEST/lora_adapters/$(basename $(dirname $d))/"
done

# Skip merged models (huge, can re-merge from adapters)
echo ""
echo "Skipped: */final/ directories (merged models, ~16GB each)"
echo "Re-merge locally with: python -c 'from peft import AutoPeftModelForCausalLM; ...'"
echo ""
echo "Done!"
