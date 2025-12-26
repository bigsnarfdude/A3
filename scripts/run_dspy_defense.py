"""
DSPy Defense Script
-------------------
This script uses DSPy with GEPA optimizer to learn optimal safety prompts from training data.
GEPA uses reflective text evolution with textual feedback from the judge model to improve
safety classification.

Usage:
    1. Ensure you have run the evaluation pipeline first to generate training/validation/ood splits
    2. Set ANTHROPIC_API_KEY and OPENROUTER_API_KEY environment variables
    3. Run the script to optimize and evaluate the safety classifier
    4. Results will be saved to dspy_defense_results_<model>/

Key Features:
    - Uses entire training set (no sampling)
    - Gets textual feedback from judge model (Claude Sonnet 4)
    - GEPA reflects on failures and proposes better prompts
    - Automatically evolves instructions based on judge reasoning
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.dspy_defense_agent import DSPyDefenseAgent, DSPyDefenseConfig  # noqa: E402
from agent.config_loader import load_attack_config  # noqa: E402


DEFAULT_MODEL = "claude-sonnet-4-20250514"


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run DSPy defense optimization with GEPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run DSPy defense with medium optimization
  python scripts/run_dspy_defense.py --config-file configs/sycophancy-llama.json

  # Use heavy optimization for thorough search
  python scripts/run_dspy_defense.py --config-file configs/sycophancy-llama.json --auto heavy

  # Quick test with only 10 training samples
  python scripts/run_dspy_defense.py --config-file configs/sycophancy-llama.json --max-train-samples 10

  # Quick experimentation with light optimization and limited samples
  python scripts/run_dspy_defense.py --config-file configs/sycophancy-llama.json --auto light --max-train-samples 20

  # Use a different model for optimization
  python scripts/run_dspy_defense.py --config-file configs/sycophancy-llama.json --model claude-opus-4-20250514

Optimization Levels:
  light:  ~10 full evaluations (quick experimentation)
  medium: ~30 full evaluations (balanced, default)
  heavy:  ~100 full evaluations (thorough)

Quick Testing:
  Use --max-train-samples to limit training data for faster testing
  Example: --max-train-samples 10 (uses only 10 training examples)
        """
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to attack config file (e.g., configs/sycophancy-llama.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for optimization (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--auto",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Optimization intensity (default: medium)"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for quick testing). If not specified, uses all training data."
    )
    args = parser.parse_args()

    # Require API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Environment variable ANTHROPIC_API_KEY is required for optimizer and judge model"
        )

    # Load config from file
    print(f"Loading attack config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Target model: {attack_config.target_model.model_name}")
    print(f"Model name for files: {model_name}")
    print(f"Optimization level: {args.auto}")
    print(f"Optimizer model: {args.model}\n")

    # Configure DSPy defense agent
    dspy_config = DSPyDefenseConfig()
    dspy_config.model_name = model_name
    dspy_config.behavior_key = behavior_key
    dspy_config.target_model = attack_config.target_model.model_name
    dspy_config.auto = args.auto
    dspy_config.optimizer_model = args.model

    print(f"DSPy Configuration:")
    print(f"  Reflection model: {dspy_config.optimizer_model}")
    print(f"  Target model: {dspy_config.target_model}")
    print(f"  Optimization mode: {dspy_config.auto}")
    print(f"  Optimizer: GEPA (Genetic-Pareto with judge feedback)")
    print(f"  Training: Uses ENTIRE training set")
    print(f"  Feedback: Textual reasoning from judge model\n")

    # Initialize agent
    agent = DSPyDefenseAgent(config=dspy_config)

    # Run DSPy defense (optimize + evaluate)
    print(f"\n{'='*100}")
    print("STARTING DSPY DEFENSE PIPELINE")
    print(f"{'='*100}\n")

    result = agent.run_dspy_defense(iteration=1, max_train_samples=args.max_train_samples)

    # Print summary
    print(f"\n{'='*100}")
    print("DSPY DEFENSE RESULTS SUMMARY")
    print(f"{'='*100}")

    print(f"\nValidation Set Performance:")
    print(f"  Harmful: {result.validation_harmful_count}/{result.validation_harmful_total} "
          f"attack success ({result.validation_harmful_rate:.1%})")
    print(f"  Benign:  {result.validation_benign_refusal_count}/{result.validation_benign_total} "
          f"refusals ({result.validation_benign_refusal_rate:.1%})")

    # Calculate validation accuracy
    val_correct = (result.validation_harmful_total - result.validation_harmful_count) + \
                  (result.validation_benign_total - result.validation_benign_refusal_count)
    val_total = result.validation_harmful_total + result.validation_benign_total
    val_accuracy = val_correct / val_total if val_total > 0 else 0.0
    print(f"  Overall Accuracy: {val_correct}/{val_total} ({val_accuracy:.1%})")

    print(f"\nOOD Set Performance:")
    print(f"  Harmful: {result.ood_harmful_count}/{result.ood_harmful_total} "
          f"attack success ({result.ood_harmful_rate:.1%})")
    print(f"  Benign:  {result.ood_benign_refusal_count}/{result.ood_benign_total} "
          f"refusals ({result.ood_benign_refusal_rate:.1%})")

    # Calculate OOD accuracy
    ood_correct = (result.ood_harmful_total - result.ood_harmful_count) + \
                  (result.ood_benign_total - result.ood_benign_refusal_count)
    ood_total = result.ood_harmful_total + result.ood_benign_total
    ood_accuracy = ood_correct / ood_total if ood_total > 0 else 0.0
    print(f"  Overall Accuracy: {ood_correct}/{ood_total} ({ood_accuracy:.1%})")

    print(f"\n{'='*100}\n")

    # Save results
    agent.save_dspy_results(result, iteration=1)

    print("\nDSPy defense pipeline completed successfully!")

    # Print interpretation guide
    print(f"\n{'='*100}")
    print("INTERPRETING RESULTS")
    print(f"{'='*100}")
    print("""
For Harmful Prompts:
  - Lower attack success rate is BETTER (means more prompts were correctly refused)
  - We want the model to REFUSE harmful requests

For Benign Prompts:
  - Lower refusal rate is BETTER (means more prompts were correctly answered)
  - We want the model to ANSWER legitimate requests

Overall Accuracy:
  - Higher is better
  - Combines correct refusals (harmful) + correct answers (benign)

The DSPy optimizer uses GEPA with judge model feedback to learn the best instructions
that maximize accuracy across both harmful and benign prompts.

GEPA reflects on program traces and uses textual feedback from the judge model
(explaining WHY responses are safe/unsafe) to propose better prompts.
    """)


if __name__ == "__main__":
    main()
