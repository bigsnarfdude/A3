#!/bin/bash
#
# A3 Pipeline Runner
# ------------------
# Runs the complete A3 pipeline: data generation, evaluation, expected behavior
# generation, and iterative SFT training.
#
# Usage:
#   ./run_pipeline.sh <config-file> [options]
#
# Example:
#   ./run_pipeline.sh configs/sycophancy-llama.json
#   ./run_pipeline.sh configs/sycophancy-llama.json --skip-sft
#   ./run_pipeline.sh configs/sycophancy-llama.json --num-iterations 20

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SKIP_DATA_GEN=false
SKIP_EVAL=false
SKIP_EXPECTED_BEHAVIORS=false
SKIP_SFT=false
NUM_ITERATIONS=30
MAX_EPOCHS=10
MODEL="claude-sonnet-4-20250514"

# Parse arguments
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data-gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-expected-behaviors)
            SKIP_EXPECTED_BEHAVIORS=true
            shift
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --num-iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <config-file> [options]"
            echo ""
            echo "Options:"
            echo "  --skip-data-gen           Skip Step 1 (data generation)"
            echo "  --skip-eval               Skip Step 2 (evaluation/splitting)"
            echo "  --skip-expected-behaviors Skip Step 3 (expected behavior generation)"
            echo "  --skip-sft                Skip Step 4 (SFT training)"
            echo "  --num-iterations N        Number of SFT iterations (default: 30)"
            echo "  --max-epochs N            Max epochs per iteration (default: 10)"
            echo "  --model MODEL             Model for data generation (default: claude-sonnet-4-20250514)"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$CONFIG_FILE" ]]; then
                CONFIG_FILE="$1"
            else
                echo -e "${RED}Error: Unknown argument: $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate config file
if [[ -z "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: Config file is required${NC}"
    echo "Usage: $0 <config-file> [options]"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check environment variables
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY environment variable is not set${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       A3 Pipeline Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Config file: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Model: ${GREEN}$MODEL${NC}"
echo -e "SFT iterations: ${GREEN}$NUM_ITERATIONS${NC}"
echo -e "Max epochs: ${GREEN}$MAX_EPOCHS${NC}"
echo ""

# Track timing
START_TIME=$(date +%s)

# Step 1: Data Generation
if [[ "$SKIP_DATA_GEN" == "false" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Step 1: Data Generation${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    python scripts/step1_data_generation.py \
        --config-file "$CONFIG_FILE" \
        --model "$MODEL"

    echo ""
    echo -e "${GREEN}Step 1 completed successfully!${NC}"
    echo ""
else
    echo -e "${BLUE}Skipping Step 1 (data generation)${NC}"
fi

# Step 2: Evaluation and Splitting
if [[ "$SKIP_EVAL" == "false" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Step 2: Evaluation and Data Splitting${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    python scripts/step2_evaluation.py \
        --config-file "$CONFIG_FILE" \
        --model "$MODEL"

    echo ""
    echo -e "${GREEN}Step 2 completed successfully!${NC}"
    echo ""
else
    echo -e "${BLUE}Skipping Step 2 (evaluation)${NC}"
fi

# Step 3: Generate Expected Behaviors
if [[ "$SKIP_EXPECTED_BEHAVIORS" == "false" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Step 3: Generate Expected Behaviors${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    python scripts/step3_generate_expected_behaviors.py \
        --config-file "$CONFIG_FILE" \
        --model "$MODEL"

    echo ""
    echo -e "${GREEN}Step 3 completed successfully!${NC}"
    echo ""
else
    echo -e "${BLUE}Skipping Step 3 (expected behaviors)${NC}"
fi

# Step 4: SFT Training
if [[ "$SKIP_SFT" == "false" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Step 4: Iterative SFT Training${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    python scripts/step4_sft_agent.py \
        --config-file "$CONFIG_FILE" \
        --num-iterations "$NUM_ITERATIONS" \
        --max-epochs "$MAX_EPOCHS"

    echo ""
    echo -e "${GREEN}Step 4 completed successfully!${NC}"
    echo ""
else
    echo -e "${BLUE}Skipping Step 4 (SFT training)${NC}"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}       Pipeline Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "Results saved to:"
echo -e "  - hypothesis_results_*/"
echo -e "  - evaluation_results_*/"
echo -e "  - experiment_logs_*/"
echo -e "  - sft_models_*/"
echo ""
