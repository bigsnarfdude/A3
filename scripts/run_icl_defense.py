"""
ICL Defense Script
------------------
This script runs iterative In-Context Learning (ICL) defense optimization.
It selects optimal examples from training data to include in the prompt context.

Usage:
    1. Ensure you have run the evaluation pipeline first to generate training/validation/ood splits
    2. Set ANTHROPIC_API_KEY environment variable
    3. Run the script to optimize and evaluate the ICL defense
    4. Results will be saved to icl_defense_results_<behavior>_<model>/

Key Features:
    - Multiple selection methods: prompt_level, hypothesis_level, random
    - Iterative optimization with different random seeds
    - Tracks performance across iterations
    - Updates experiment log with results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.icl_defense_agent import ICLDefenseAgent, ICLDefenseConfig  # noqa: E402
from agent.config_loader import load_attack_config  # noqa: E402


DEFAULT_MODEL = "claude-sonnet-4-20250514"


def main() -> None:
    # Require Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Environment variable ANTHROPIC_API_KEY is required"
        )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run multiple iterations of ICL defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 5 iterations with prompt_level selection
  python scripts/run_icl_defense.py --config-file configs/sycophancy-llama.json

  # Run with hypothesis-level selection
  python scripts/run_icl_defense.py --config-file configs/sycophancy-llama.json \\
    --selection-method hypothesis_level --trial 2

  # Run 10 iterations with 30 examples each
  python scripts/run_icl_defense.py --config-file configs/sycophancy-llama.json \\
    --num-iterations 10 --num-icl-examples 30

  # Use a different model
  python scripts/run_icl_defense.py --config-file configs/sycophancy-llama.json \\
    --model claude-opus-4-20250514
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
        help=f"Model to use for ICL defense (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="prompt_level",
        choices=["prompt_level", "hypothesis_level", "random"],
        help="Selection method to use (default: prompt_level)"
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Trial number for this run (default: 1)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of iterations to run (default: 5)"
    )
    parser.add_argument(
        "--num-icl-examples",
        type=int,
        default=20,
        help="Number of ICL examples to select per iteration (default: 20)"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Target model: {attack_config.target_model.model_name}")
    print(f"Model name for files: {model_name}")
    print(f"ICL model: {args.model}")
    print(f"Selection method: {args.selection_method}")
    print(f"Trial: {args.trial}")
    print(f"Iterations: {args.num_iterations}")
    print(f"ICL examples per iteration: {args.num_icl_examples}\n")

    # Configuration
    num_iterations = args.num_iterations
    num_icl_examples = args.num_icl_examples

    # Store results from all iterations
    all_results = []

    print(f"\n{'='*100}")
    print(f"RUNNING {num_iterations} ITERATIONS OF ICL DEFENSE")
    print(f"Selection Method: {args.selection_method}, Trial: {args.trial}")
    print(f"{'='*100}\n")

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'#'*100}")
        print(f"# ITERATION {iteration}/{num_iterations}")
        print(f"{'#'*100}\n")

        # Use different random seed for each iteration to get diverse selections
        config = ICLDefenseConfig(
            model_name=model_name,
            behavior_key=behavior_key,
            num_icl_examples=num_icl_examples,
            random_seed=42 + iteration + (args.trial * 100),  # Different seed per trial and iteration
            selection_method=args.selection_method,
            icl_model=args.model
        )
        agent = ICLDefenseAgent(config=config)

        # Run ICL defense
        result = agent.run_icl_defense(iteration=iteration)

        # Print iteration summary
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} SUMMARY")
        print(f"{'='*80}")
        print(f"\nValidation Set:")
        print(f"  Harmful: {result.validation_harmful_count}/{result.validation_harmful_total} "
              f"attack success ({result.validation_harmful_rate:.1%})")
        print(f"  Benign:  {result.validation_benign_harmful_count}/{result.validation_benign_total} "
              f"refusals ({result.validation_benign_harmful_rate:.1%})")

        print(f"\nOOD Set:")
        print(f"  Harmful: {result.ood_harmful_count}/{result.ood_harmful_total} "
              f"attack success ({result.ood_harmful_rate:.1%})")
        print(f"  Benign:  {result.ood_benign_harmful_count}/{result.ood_benign_total} "
              f"refusals ({result.ood_benign_harmful_rate:.1%})")
        print(f"{'='*80}\n")

        # Save results
        agent.save_icl_results(result, iteration=iteration)

        # Update experiment log
        agent.update_experiment_log(result, iteration=iteration)

        # Store result for final summary
        all_results.append({
            "iteration": iteration,
            "validation_harmful_rate": result.validation_harmful_rate,
            "validation_benign_harmful_rate": result.validation_benign_harmful_rate,
            "ood_harmful_rate": result.ood_harmful_rate,
            "ood_benign_harmful_rate": result.ood_benign_harmful_rate,
            "num_icl_examples": len(result.icl_examples)
        })

        # Small delay between iterations
        if iteration < num_iterations:
            print(f"\nWaiting 2 seconds before next iteration...\n")
            time.sleep(2)

    # Print final summary across all iterations
    print(f"\n{'='*100}")
    print("FINAL SUMMARY ACROSS ALL ITERATIONS")
    print(f"{'='*100}\n")

    print(f"{'Iteration':<12} {'Val Harmful':<15} {'Val Benign':<15} {'OOD Harmful':<15} {'OOD Benign':<15}")
    print("-" * 100)

    for result_summary in all_results:
        iteration = result_summary["iteration"]
        val_harm = result_summary["validation_harmful_rate"]
        val_benign = result_summary["validation_benign_harmful_rate"]
        ood_harm = result_summary["ood_harmful_rate"]
        ood_benign = result_summary["ood_benign_harmful_rate"]

        print(f"{iteration:<12} {val_harm:<15.1%} {val_benign:<15.1%} {ood_harm:<15.1%} {ood_benign:<15.1%}")

    # Calculate averages
    avg_val_harm = sum(r["validation_harmful_rate"] for r in all_results) / len(all_results)
    avg_val_benign = sum(r["validation_benign_harmful_rate"] for r in all_results) / len(all_results)
    avg_ood_harm = sum(r["ood_harmful_rate"] for r in all_results) / len(all_results)
    avg_ood_benign = sum(r["ood_benign_harmful_rate"] for r in all_results) / len(all_results)

    print("-" * 100)
    print(f"{'AVERAGE':<12} {avg_val_harm:<15.1%} {avg_val_benign:<15.1%} {avg_ood_harm:<15.1%} {avg_ood_benign:<15.1%}")

    print(f"\n{'='*100}")
    print("INTERPRETATION:")
    print("- Lower harmful rates are better (fewer successful attacks)")
    print("- Lower benign refusal rates are better (fewer false positives)")
    print(f"{'='*100}\n")

    print(f"\nAll {num_iterations} iterations completed successfully!")
    results_dir_name = f"icl_defense_results_{behavior_key}_{model_name}"
    logs_dir_name = f"experiment_logs_{behavior_key}_{model_name}"
    print(f"Results saved in {results_dir_name}/ directory")
    print(f"Experiment log updated in {logs_dir_name}/experiment_log_latest.txt")


if __name__ == "__main__":
    main()
