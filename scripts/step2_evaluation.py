"""
Step 2: Evaluation
------------------
Creates training/validation/OOD splits from generated hypotheses and evaluates
the target model on validation and OOD sets to measure defense effectiveness.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_agent import EvaluationAgent, EvalAgentConfig  # noqa: E402
from agent.config_loader import load_attack_config  # noqa: E402


def load_hypotheses_from_results(model_name: str, behavior_key: str = None) -> List[Dict[str, Any]]:
    """Load all hypothesis information from saved results files."""
    if behavior_key:
        results_dir = Path(f"hypothesis_results_{behavior_key}_{model_name}")
    else:
        results_dir = Path(f"hypothesis_results_{model_name}")
    if not results_dir.exists():
        raise RuntimeError(f"{results_dir} directory not found. Run step1_data_generation.py first.")

    # Group files by hypothesis index
    hypothesis_files: Dict[int, List[Path]] = {}
    for file_path in results_dir.glob("hypothesis_*.json"):
        parts = file_path.stem.split("_")
        if len(parts) >= 2 and parts[0] == "hypothesis":
            try:
                h_idx = int(parts[1])
                if h_idx not in hypothesis_files:
                    hypothesis_files[h_idx] = []
                hypothesis_files[h_idx].append(file_path)
            except ValueError:
                continue

    # Load the most recent file for each hypothesis
    hypotheses_info = []
    for h_idx in sorted(hypothesis_files.keys()):
        files = hypothesis_files[h_idx]
        latest_file = max(files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r') as f:
            data = json.load(f)
            hypotheses_info.append({
                'index': h_idx,
                'text': data.get('hypothesis_text', ''),
                'num_harmful_prompts': data.get('num_harmful_prompts', 0),
                'num_benign_prompts': data.get('num_benign_prompts', 0),
                'summary': data.get('summary', {})
            })

    return hypotheses_info


def save_evaluation_results(results, model_name: str, behavior_key: str = None, output_dir: Path | None = None):
    """Save evaluation results and data splits to disk."""
    if output_dir is None:
        if behavior_key:
            output_dir = Path(f"evaluation_results_{behavior_key}_{model_name}")
        else:
            output_dir = Path(f"evaluation_results_{model_name}")
    output_dir.mkdir(exist_ok=True)

    # Save summary
    summary = {
        "validation": {
            "harmful_prompts": {
                "harmful_count": results.validation_harmful_count,
                "total": results.validation_harmful_total,
                "harmful_rate": results.validation_harmful_rate
            },
            "benign_prompts": {
                "harmful_count": results.validation_benign_harmful_count,
                "total": results.validation_benign_total,
                "harmful_rate": results.validation_benign_harmful_rate
            }
        },
        "ood": {
            "harmful_prompts": {
                "harmful_count": results.ood_harmful_count,
                "total": results.ood_harmful_total,
                "harmful_rate": results.ood_harmful_rate
            },
            "benign_prompts": {
                "harmful_count": results.ood_benign_harmful_count,
                "total": results.ood_benign_total,
                "harmful_rate": results.ood_benign_harmful_rate
            }
        }
    }

    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save data splits
    for split_name, split_data in [
        ("training", results.training_split),
        ("validation", results.validation_split),
        ("ood", results.ood_split)
    ]:
        split_file = output_dir / f"{split_name}_split.json"
        split_dict = {
            "hypothesis_indices": split_data.hypothesis_indices,
            "split_reasoning": split_data.split_reasoning,
            "harmful_prompts": {
                "num_prompts": len(split_data.harmful_prompts),
                "prompts": split_data.harmful_prompts,
                "labels": split_data.harmful_labels
            },
            "benign_prompts": {
                "num_prompts": len(split_data.benign_prompts),
                "prompts": split_data.benign_prompts,
                "labels": split_data.benign_labels
            }
        }
        with open(split_file, 'w') as f:
            json.dump(split_dict, f, indent=2)

    print(f"\nSaved evaluation results to {output_dir}/")


def save_experiment_log_init(behavior_key: str, model_name: str) -> None:
    """Copy experiment_log_latest.txt to experiment_log_init.txt for baseline preservation."""
    log_dir = Path(f"experiment_logs_{behavior_key}_{model_name}")
    latest_log = log_dir / "experiment_log_latest.txt"
    init_log = log_dir / "experiment_log_init.txt"

    if latest_log.exists():
        shutil.copy2(latest_log, init_log)
        print(f"Saved initial experiment log to {init_log}")
    else:
        print(f"Warning: {latest_log} not found, skipping init log save")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create training/validation/OOD splits and evaluate target model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate sycophancy attack on Llama
  python scripts/step2_evaluation.py --config-file configs/sycophancy-llama.json

  # Use a different model for evaluation
  python scripts/step2_evaluation.py --config-file configs/sycophancy-llama.json \\
    --model claude-opus-4-20250514
        """
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to attack config JSON file (e.g., configs/sycophancy-llama.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for data splitting and evaluation (default: claude-sonnet-4-20250514)"
    )
    args = parser.parse_args()

    # Require Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Environment variable ANTHROPIC_API_KEY is required"
        )

    # Load config from file
    print(f"Loading attack config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Target model: {attack_config.target_model.model_name}")
    print(f"Model name for files: {model_name}")
    print(f"Output format: {attack_config.conversation_format.output_format}")
    if attack_config.conversation_format.output_format == "conversation":
        print(f"  Conversation turns: {attack_config.conversation_format.num_turns}")
        print(f"  Include system prompt: {attack_config.conversation_format.include_system_prompt}")
    print()

    # Load all hypothesis information from saved results
    print("Loading hypothesis information from saved results...")
    hypotheses_info = load_hypotheses_from_results(model_name, behavior_key)

    if not hypotheses_info:
        raise RuntimeError("No hypothesis results found. Run step1_data_generation.py first.")

    print(f"Found {len(hypotheses_info)} hypotheses")
    for h_info in hypotheses_info:
        total_prompts = h_info.get('num_harmful_prompts', 0) + h_info.get('num_benign_prompts', 0)
        print(f"  H {h_info['index']}: {total_prompts} prompts ({h_info.get('num_harmful_prompts', 0)} harmful + {h_info.get('num_benign_prompts', 0)} benign) - {h_info['text'][:80]}...")

    # Initialize evaluation agent with model name, behavior, and attack config
    config = EvalAgentConfig(model_name=model_name, behavior_key=behavior_key)
    agent = EvaluationAgent(config=config, eval_model=args.model, attack_config=attack_config)

    print(f"\nEvaluation model: {args.model}")
    print(f"Target model for evaluation: {attack_config.target_model.model_name}")
    print(f"Judge model: {attack_config.judge_model.model_name}")

    # Create splits and evaluate
    results = agent.create_splits_and_evaluate(hypotheses_info)

    # Print results
    print(f"\n{'='*100}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*100}")
    print(f"Training Set: {len(results.training_split.harmful_prompts)} harmful + {len(results.training_split.benign_prompts)} benign prompts from {len(results.training_split.hypothesis_indices)} hypotheses")
    print(f"  Hypotheses: {results.training_split.hypothesis_indices}")
    print(f"\nValidation Set:")
    print(f"  Harmful prompts: {results.validation_harmful_count}/{results.validation_harmful_total} attack success ({results.validation_harmful_rate:.1%})")
    print(f"  Benign prompts:  {results.validation_benign_harmful_count}/{results.validation_benign_total} refusals ({results.validation_benign_harmful_rate:.1%})")
    print(f"\nOOD Evaluation Set: {len(results.ood_split.harmful_prompts)} harmful + {len(results.ood_split.benign_prompts)} benign prompts from {len(results.ood_split.hypothesis_indices)} hypotheses")
    print(f"  Hypotheses: {results.ood_split.hypothesis_indices}")
    print(f"  Harmful prompts: {results.ood_harmful_count}/{results.ood_harmful_total} attack success ({results.ood_harmful_rate:.1%})")
    print(f"  Benign prompts:  {results.ood_benign_harmful_count}/{results.ood_benign_total} refusals ({results.ood_benign_harmful_rate:.1%})")
    print(f"{'='*100}\n")

    # Save results
    save_evaluation_results(results, model_name, behavior_key)

    # Save initial experiment log for baseline preservation
    save_experiment_log_init(behavior_key, model_name)


if __name__ == "__main__":
    main()
