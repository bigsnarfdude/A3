"""
SFT Training Script with Fixed Data Mixing Configurations
-----------------------------------------------------------
Runs SFT training with known, fixed data mixing ratios for ALL configurations:
- Always 15% Dolci dataset mixing for general instruction capability retention
- Remaining 85% split between harmful and benign samples

Loops over ALL five harmful/benign ratios (for the 85% non-Dolci portion):
- 10/90: 10% harmful, 90% benign
- 30/70: 30% harmful, 70% benign
- 50/50: 50% harmful, 50% benign
- 70/30: 70% harmful, 30% benign
- 90/10: 90% harmful, 10% benign

Default Configuration:
- LoRA: Enabled (r=64, alpha=256)
- Epochs: 5
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Learning rate: 1e-5
- Dolci mixing: 15% (fixed)

Usage:
    # Run all 5 configurations sequentially (uses LoRA by default)
    python scripts/run_sft_with_fixed_mixing.py --config-file configs/sycophancy-llama.json

    # Disable LoRA for full fine-tuning
    python scripts/run_sft_with_fixed_mixing.py --config-file configs/sycophancy-llama.json --no-lora

    # Custom model path
    python scripts/run_sft_with_fixed_mixing.py --config-file configs/sycophancy-llama.json \
        --model-name-or-path meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.sft_agent import SFTAgent, SFTConfig, SFTResult
from agent.config_loader import AttackConfig
from agent.evaluation_agent import DataSplit

# Fixed Dolci percentage
DOLCI_PERCENTAGE = 15.0

# Supported harmful/benign ratios
SUPPORTED_RATIOS = [10, 30, 50, 70, 90]


def load_data_split(split_file: Path) -> DataSplit:
    """Load a data split from JSON file."""
    with open(split_file, "r") as f:
        data = json.load(f)

    return DataSplit(
        hypothesis_indices=data.get("hypothesis_indices", []),
        harmful_prompts=data["harmful_prompts"]["prompts"],
        harmful_labels=data["harmful_prompts"]["labels"],
        benign_prompts=data["benign_prompts"]["prompts"],
        benign_labels=data["benign_prompts"]["labels"],
        split_reasoning=data.get("split_reasoning")
    )


def load_expected_behaviors(behaviors_file: Path) -> dict:
    """Load expected behaviors from JSON file."""
    with open(behaviors_file, "r") as f:
        data = json.load(f)
    return data


def prepare_training_data_with_fixed_mixing(
    training_split: DataSplit,
    expected_behaviors: Dict[str, str],
    harmful_ratio: int,
    target_size: int,
    dolci_responses_file: Optional[str] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Prepare training data with fixed mixing ratios.

    Args:
        training_split: Training data split
        expected_behaviors: Dict mapping prompts to expected behaviors
        harmful_ratio: Percentage of harmful samples in the non-Dolci portion (10, 30, 50, 70, 90)
        target_size: Total target dataset size
        dolci_responses_file: Optional path to pre-generated responses for Dolci
        seed: Random seed for reproducibility

    Returns:
        List of training examples in messages format
    """
    np.random.seed(seed)

    # Calculate sample counts
    dolci_count = int(target_size * (DOLCI_PERCENTAGE / 100.0))
    non_dolci_count = target_size - dolci_count

    harmful_count = int(non_dolci_count * (harmful_ratio / 100.0))
    benign_count = non_dolci_count - harmful_count

    print(f"\nData mixing configuration:")
    print(f"  Total target size: {target_size}")
    print(f"  Dolci samples: {dolci_count} ({DOLCI_PERCENTAGE:.1f}%)")
    print(f"  Non-Dolci samples: {non_dolci_count} ({100-DOLCI_PERCENTAGE:.1f}%)")
    print(f"    Harmful: {harmful_count} ({harmful_ratio}% of non-Dolci)")
    print(f"    Benign: {benign_count} ({100-harmful_ratio}% of non-Dolci)")

    training_data = []

    # Prepare harmful samples
    harmful_samples = []
    for prompt in training_split.harmful_prompts:
        prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
        expected = expected_behaviors.get(
            prompt_key,
            "I cannot and will not provide assistance with that request."
        )
        harmful_samples.append({
            "prompt": prompt,
            "expected": expected
        })

    # Prepare benign samples
    benign_samples = []
    for prompt in training_split.benign_prompts:
        prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
        expected = expected_behaviors.get(prompt_key)
        if expected:
            benign_samples.append({
                "prompt": prompt,
                "expected": expected
            })

    print(f"\nAvailable samples:")
    print(f"  Harmful: {len(harmful_samples)}")
    print(f"  Benign: {len(benign_samples)}")

    # Sample harmful prompts (with replacement if needed)
    if harmful_count > 0 and len(harmful_samples) > 0:
        harmful_indices = np.random.choice(
            len(harmful_samples),
            size=harmful_count,
            replace=harmful_count > len(harmful_samples)
        )
        for idx in harmful_indices:
            sample = harmful_samples[idx]
            # Skip if expected response is None
            if sample["expected"] is None:
                continue
            if isinstance(sample["prompt"], list):
                # Validate conversation messages have non-None content
                valid = all(msg.get("content") is not None for msg in sample["prompt"])
                if not valid:
                    continue
                messages = sample["prompt"] + [{"role": "assistant", "content": sample["expected"]}]
            else:
                if sample["prompt"] is None:
                    continue
                messages = [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample["expected"]}
                ]
            training_data.append({"messages": messages})

    # Sample benign prompts (with replacement if needed)
    if benign_count > 0 and len(benign_samples) > 0:
        benign_indices = np.random.choice(
            len(benign_samples),
            size=benign_count,
            replace=benign_count > len(benign_samples)
        )
        for idx in benign_indices:
            sample = benign_samples[idx]
            # Skip if expected response is None
            if sample["expected"] is None:
                continue
            if isinstance(sample["prompt"], list):
                # Validate conversation messages have non-None content
                valid = all(msg.get("content") is not None for msg in sample["prompt"])
                if not valid:
                    continue
                messages = sample["prompt"] + [{"role": "assistant", "content": sample["expected"]}]
            else:
                if sample["prompt"] is None:
                    continue
                messages = [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample["expected"]}
                ]
            training_data.append({"messages": messages})

    # Add Dolci samples
    dolci_sampled_count = 0
    if dolci_count > 0:
        if dolci_responses_file and Path(dolci_responses_file).exists():
            print(f"\nLoading pre-generated responses from: {dolci_responses_file}")
            try:
                with open(dolci_responses_file, 'r') as f:
                    dolci_responses_data = json.load(f)

                print(f"Loaded {len(dolci_responses_data)} pre-generated responses")

                if len(dolci_responses_data) < dolci_count:
                    print(f"Warning: DOLCI responses file has only {len(dolci_responses_data)} examples, requested {dolci_count}")
                    dolci_count = len(dolci_responses_data)

                dolci_indices = np.random.choice(len(dolci_responses_data), size=dolci_count, replace=False)

                for idx in dolci_indices:
                    dolci_sample = dolci_responses_data[int(idx)]
                    messages = dolci_sample.get('messages', [])
                    # Try model_response first, fall back to qwen_response for backward compat
                    model_response = dolci_sample.get('model_response') or dolci_sample.get('qwen_response', '')

                    if not messages or not model_response:
                        continue

                    # Validate that all messages have non-None content
                    valid = True
                    for msg in messages:
                        if msg.get("content") is None:
                            valid = False
                            break
                    if not valid:
                        continue

                    full_messages = messages + [{"role": "assistant", "content": model_response}]
                    training_data.append({"messages": full_messages})
                    dolci_sampled_count += 1

                print(f"Added {dolci_sampled_count} Dolci samples with pre-generated responses")

            except Exception as e:
                print(f"Warning: Failed to load DOLCI responses file: {e}")
                print("Falling back to original Dolci dataset...")
                dolci_responses_file = None

        if not dolci_responses_file or not Path(dolci_responses_file).exists():
            print(f"\nLoading Dolci-Instruct-SFT dataset from HuggingFace...")
            try:
                from datasets import load_dataset

                dolci_dataset = load_dataset("allenai/Dolci-Instruct-SFT", split="train")

                if len(dolci_dataset) < dolci_count:
                    print(f"Warning: Dolci dataset has only {len(dolci_dataset)} examples, requested {dolci_count}")
                    dolci_count = len(dolci_dataset)

                dolci_indices = np.random.choice(len(dolci_dataset), size=dolci_count, replace=False)

                for idx in dolci_indices:
                    dolci_sample = dolci_dataset[int(idx)]

                    if 'messages' in dolci_sample:
                        messages = dolci_sample['messages']
                    elif 'conversations' in dolci_sample:
                        messages = dolci_sample['conversations']
                    elif 'prompt' in dolci_sample and 'response' in dolci_sample:
                        messages = [
                            {"role": "user", "content": dolci_sample['prompt']},
                            {"role": "assistant", "content": dolci_sample['response']}
                        ]
                    else:
                        continue

                    # Validate that all messages have non-None content
                    valid = True
                    for msg in messages:
                        if msg.get("content") is None:
                            valid = False
                            break
                    if not valid:
                        continue

                    training_data.append({"messages": messages})
                    dolci_sampled_count += 1

                print(f"Added {dolci_sampled_count} Dolci samples")

            except Exception as e:
                print(f"Warning: Failed to load Dolci dataset: {e}")
                print("Continuing with only harmful/benign samples...")

    # Shuffle
    np.random.shuffle(training_data)

    actual_harmful = harmful_count
    actual_benign = benign_count
    actual_dolci = dolci_sampled_count

    print(f"\nFinal dataset composition:")
    print(f"  Total: {len(training_data)}")
    print(f"  Harmful: {actual_harmful} ({actual_harmful/len(training_data)*100:.1f}%)")
    print(f"  Benign: {actual_benign} ({actual_benign/len(training_data)*100:.1f}%)")
    print(f"  Dolci: {actual_dolci} ({actual_dolci/len(training_data)*100:.1f}%)")

    return training_data


def run_single_configuration(
    harmful_ratio: int,
    training_split: DataSplit,
    validation_split: DataSplit,
    ood_split: DataSplit,
    expected_behaviors: Dict[str, str],
    attack_config: Any,
    model_name: str,
    behavior_key: str,
    args: argparse.Namespace,
    target_size: int,
    dolci_responses_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run training and evaluation for a single harmful/benign ratio configuration.

    Returns:
        Dictionary containing results for this configuration
    """
    print(f"\n{'#'*100}")
    print(f"# CONFIGURATION: {harmful_ratio}% harmful / {100-harmful_ratio}% benign (+ {DOLCI_PERCENTAGE}% Dolci)")
    print(f"{'#'*100}\n")

    # Configure SFT agent
    sft_config = SFTConfig()
    sft_config.model_name_or_path = args.model_name_or_path
    sft_config.model_name = model_name
    sft_config.behavior_key = behavior_key
    sft_config.num_train_epochs = args.epochs
    sft_config.per_device_train_batch_size = args.batch_size
    sft_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    sft_config.learning_rate = args.learning_rate

    # Configure benchmark evaluation
    sft_config.enable_benchmark_eval = not args.disable_benchmark_eval
    sft_config.mmlu_pro_subset_size = args.mmlu_pro_subset_size
    sft_config.gpqa_subset_size = args.gpqa_subset_size

    # Configure LoRA
    sft_config.use_lora = args.use_lora
    sft_config.lora_r = args.lora_r
    sft_config.lora_alpha = args.lora_alpha
    sft_config.lora_dropout = args.lora_dropout

    # Create output directory with mixing config in name
    output_suffix = f"dolci{int(DOLCI_PERCENTAGE)}_harmful{harmful_ratio}_benign{100-harmful_ratio}"
    output_dir = f"./sft_models_{behavior_key}_{model_name}_{output_suffix}"

    print(f"{'='*80}")
    print("SFT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {sft_config.model_name_or_path}")
    print(f"Epochs: {sft_config.num_train_epochs}")
    print(f"Batch size per device: {sft_config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {sft_config.gradient_accumulation_steps}")
    print(f"Learning rate: {sft_config.learning_rate}")
    print(f"LoRA enabled: {sft_config.use_lora}")
    if sft_config.use_lora:
        print(f"  LoRA rank (r): {sft_config.lora_r}")
        print(f"  LoRA alpha: {sft_config.lora_alpha}")
        print(f"  LoRA dropout: {sft_config.lora_dropout}")
    print(f"\nData Mixing:")
    print(f"  Dolci: {DOLCI_PERCENTAGE}% (fixed)")
    print(f"  Harmful: {harmful_ratio}% of non-Dolci")
    print(f"  Benign: {100-harmful_ratio}% of non-Dolci")
    print(f"  Target dataset size: {target_size}")
    print(f"  Random seed: {args.seed}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Benchmark evaluation: {sft_config.enable_benchmark_eval}")
    print(f"{'='*80}\n")

    # Prepare training data with fixed mixing
    print(f"{'='*80}")
    print("PREPARING TRAINING DATA")
    print(f"{'='*80}")

    training_data = prepare_training_data_with_fixed_mixing(
        training_split,
        expected_behaviors,
        harmful_ratio=harmful_ratio,
        target_size=target_size,
        dolci_responses_file=dolci_responses_file,
        seed=args.seed
    )

    # Save training data to file
    data_dir = Path(f"sft_data_{behavior_key}_{model_name}_{output_suffix}")
    data_dir.mkdir(parents=True, exist_ok=True)
    training_data_file = data_dir / "training_data.json"

    with open(training_data_file, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nSaved training data to: {training_data_file}")

    # Initialize SFT agent
    sft_agent = SFTAgent(config=sft_config)

    # Train model
    print(f"\n{'='*80}")
    print("STARTING SFT TRAINING")
    print(f"{'='*80}\n")

    checkpoint_path = sft_agent.train_model(str(training_data_file), output_dir=output_dir)

    # Prepare final model path
    final_model_path = f"{checkpoint_path}/final"

    # Handle LoRA vs full fine-tuning
    if sft_config.use_lora:
        if not Path(final_model_path).exists():
            raise RuntimeError(
                f"LoRA enabled but final model not found at {final_model_path}!"
            )
        print(f"\nUsing merged LoRA model: {final_model_path}")
    else:
        import glob
        import shutil
        checkpoint_dirs = glob.glob(f"{checkpoint_path}/checkpoint-*")

        if not checkpoint_dirs:
            if Path(final_model_path).exists():
                print(f"Using existing final model: {final_model_path}")
            else:
                raise RuntimeError(f"No checkpoints found in {checkpoint_path}!")
        else:
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = checkpoint_dirs[-1]

            if Path(final_model_path).exists():
                shutil.rmtree(final_model_path)

            print(f"Copying {latest_checkpoint} -> {final_model_path}")
            shutil.copytree(latest_checkpoint, final_model_path)
            print(f"Final model ready: {final_model_path}")

    # Run benchmark evaluation
    benchmark_results_list = []
    if sft_config.enable_benchmark_eval:
        print(f"\n{'='*80}")
        print("RUNNING BENCHMARK EVALUATION")
        print(f"{'='*80}\n")

        try:
            benchmark_result = sft_agent.evaluate_benchmarks(
                str(final_model_path),
                epoch=sft_config.num_train_epochs
            )
            benchmark_results_list.append(benchmark_result)

            print(f"\nBenchmark Results:")
            print(f"  MMLU-Pro: {benchmark_result.mmlu_pro_accuracy:.2%}")
            print(f"  GPQA: {benchmark_result.gpqa_accuracy:.2%}")
            print(f"  Overall: {benchmark_result.overall_score:.2%}")

        except Exception as e:
            print(f"Error evaluating benchmarks: {e}")
            import traceback
            traceback.print_exc()

    # Start vLLM server for evaluation
    sft_agent.start_vllm_server(final_model_path)

    # Evaluate on validation and OOD sets
    validation_results, ood_results = sft_agent.evaluate_model(
        validation_split,
        ood_split,
        attack_config
    )

    # Terminate vLLM after evaluation
    sft_agent.terminate_vllm()

    # Calculate metrics
    val_harmful = validation_results["harmful"]
    val_harmful_count = sum(1 for r in val_harmful if r.get("harmful", False))
    val_harmful_total = len(val_harmful)
    val_harmful_rate = val_harmful_count / val_harmful_total if val_harmful_total > 0 else 0.0

    val_benign = validation_results["benign"]
    val_benign_refusal_count = sum(1 for r in val_benign if r.get("refused", False))
    val_benign_total = len(val_benign)
    val_benign_refusal_rate = val_benign_refusal_count / val_benign_total if val_benign_total > 0 else 0.0

    ood_harmful = ood_results["harmful"]
    ood_harmful_count = sum(1 for r in ood_harmful if r.get("harmful", False))
    ood_harmful_total = len(ood_harmful)
    ood_harmful_rate = ood_harmful_count / ood_harmful_total if ood_harmful_total > 0 else 0.0

    ood_benign = ood_results["benign"]
    ood_benign_refusal_count = sum(1 for r in ood_benign if r.get("refused", False))
    ood_benign_total = len(ood_benign)
    ood_benign_refusal_rate = ood_benign_refusal_count / ood_benign_total if ood_benign_total > 0 else 0.0

    result = SFTResult(
        training_complete=True,
        checkpoint_path=final_model_path,
        validation_harmful_rate=val_harmful_rate,
        validation_harmful_total=val_harmful_total,
        validation_harmful_count=val_harmful_count,
        validation_benign_refusal_rate=val_benign_refusal_rate,
        validation_benign_total=val_benign_total,
        validation_benign_refusal_count=val_benign_refusal_count,
        ood_harmful_rate=ood_harmful_rate,
        ood_harmful_total=ood_harmful_total,
        ood_harmful_count=ood_harmful_count,
        ood_benign_refusal_rate=ood_benign_refusal_rate,
        ood_benign_total=ood_benign_total,
        ood_benign_refusal_count=ood_benign_refusal_count,
        benchmark_results=benchmark_results_list
    )

    # Save results
    results_dir = Path(f"sft_results_{behavior_key}_{model_name}_{output_suffix}")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "model_name": model_name,
        "behavior_key": behavior_key,
        "mixing_config": {
            "dolci_percentage": DOLCI_PERCENTAGE,
            "harmful_ratio": harmful_ratio,
            "benign_ratio": 100 - harmful_ratio,
            "target_size": target_size,
            "seed": args.seed
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "use_lora": args.use_lora
        },
        "checkpoint_path": result.checkpoint_path,
        "validation": {
            "harmful_rate": result.validation_harmful_rate,
            "harmful_total": result.validation_harmful_total,
            "harmful_count": result.validation_harmful_count,
            "benign_refusal_rate": result.validation_benign_refusal_rate,
            "benign_total": result.validation_benign_total,
            "benign_refusal_count": result.validation_benign_refusal_count
        },
        "ood": {
            "harmful_rate": result.ood_harmful_rate,
            "harmful_total": result.ood_harmful_total,
            "harmful_count": result.ood_harmful_count,
            "benign_refusal_rate": result.ood_benign_refusal_rate,
            "benign_total": result.ood_benign_total,
            "benign_refusal_count": result.ood_benign_refusal_count
        },
        "benchmarks": [
            {
                "epoch": br.epoch,
                "checkpoint_path": br.checkpoint_path,
                "mmlu_pro_accuracy": br.mmlu_pro_accuracy,
                "mmlu_pro_num_questions": br.mmlu_pro_num_questions,
                "gpqa_accuracy": br.gpqa_accuracy,
                "gpqa_num_questions": br.gpqa_num_questions,
                "overall_score": br.overall_score
            }
            for br in (result.benchmark_results or [])
        ]
    }

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print configuration summary
    print(f"\n{'='*80}")
    print(f"CONFIGURATION COMPLETE: {harmful_ratio}% harmful / {100-harmful_ratio}% benign")
    print(f"{'='*80}")
    print(f"Model: {result.checkpoint_path}")
    print(f"Validation - Harmful ASR: {result.validation_harmful_rate:.1%}, Benign Refusal: {result.validation_benign_refusal_rate:.1%}")
    print(f"OOD - Harmful ASR: {result.ood_harmful_rate:.1%}, Benign Refusal: {result.ood_benign_refusal_rate:.1%}")
    if result.benchmark_results:
        for br in result.benchmark_results:
            print(f"Benchmarks - MMLU-Pro: {br.mmlu_pro_accuracy:.2%}, GPQA: {br.gpqa_accuracy:.2%}")
    print(f"{'='*80}\n")

    return results_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SFT training with fixed data mixing ratios for ALL configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 5 configurations (10/90, 30/70, 50/50, 70/30, 90/10)
  python scripts/run_sft_with_fixed_mixing.py --config-file configs/sycophancy-llama.json

  # Run with LoRA for all configurations
  python scripts/run_sft_with_fixed_mixing.py --config-file configs/sycophancy-llama.json --use-lora

  # Custom model path
  python scripts/run_sft_with_fixed_mixing.py --config-file configs/sycophancy-llama.json \
      --model-name-or-path meta-llama/Llama-3.1-8B-Instruct

Data Mixing:
  - Dolci percentage is always fixed at 15%
  - Script loops over ALL 5 harmful/benign ratios: 10, 30, 50, 70, 90
  - Each ratio applies to the remaining 85% (non-Dolci) portion
        """
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to attack config file (e.g., configs/sycophancy-llama.json)"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="HuggingFace model name or path. If not specified, inferred from config."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device (default: 2)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Target training dataset size. If not provided, uses min(harmful, benign) * 2"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--expected-behaviors-file",
        type=str,
        default=None,
        help="Path to expected behaviors JSON file"
    )
    parser.add_argument(
        "--enable-benchmark-eval",
        action="store_true",
        default=True,
        help="Enable MMLU-Pro and GPQA evaluation (default: True)"
    )
    parser.add_argument(
        "--disable-benchmark-eval",
        action="store_true",
        help="Disable MMLU-Pro and GPQA evaluation"
    )
    parser.add_argument(
        "--mmlu-pro-subset-size",
        type=int,
        default=200,
        help="Number of MMLU-Pro questions to evaluate (default: 200)"
    )
    parser.add_argument(
        "--gpqa-subset-size",
        type=int,
        default=448,
        help="GPQA subset size (default: 448, full GPQA Main)"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning (default: True)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and use full fine-tuning"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank (default: 64)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=256,
        help="LoRA alpha scaling parameter (default: 256)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0,
        help="LoRA dropout rate (default: 0)"
    )
    args = parser.parse_args()

    # Handle --no-lora flag
    if args.no_lora:
        args.use_lora = False

    # Load config from file
    print(f"Loading attack config from: {args.config_file}")
    attack_config = AttackConfig.from_file(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Target model: {attack_config.target_model.model_name}")
    print(f"Model name for files: {model_name}")

    # Determine model path: CLI override > config huggingface_model_path
    if args.model_name_or_path:
        model_path = args.model_name_or_path
    else:
        # Use huggingface_model_path from config
        try:
            model_path = attack_config.target_model.get_huggingface_path()
        except ValueError as e:
            print(f"Error: {e}")
            print("You can also specify --model-name-or-path on the command line")
            sys.exit(1)

    args.model_name_or_path = model_path
    print(f"Using model: {model_path}")

    # Get dolci_responses_file from config
    dolci_responses_file = None
    if hasattr(attack_config.paths, 'dolci_responses_file') and attack_config.paths.dolci_responses_file:
        dolci_responses_file = attack_config.paths.dolci_responses_file
        # Resolve path relative to project root
        if not Path(dolci_responses_file).is_absolute():
            dolci_responses_file = str(PROJECT_ROOT / dolci_responses_file)
        if Path(dolci_responses_file).exists():
            print(f"Using DOLCI responses file: {dolci_responses_file}")
        else:
            print(f"Warning: DOLCI responses file not found: {dolci_responses_file}")
            print("  Will fall back to downloading from HuggingFace")
            dolci_responses_file = None

    # Find evaluation results directory
    eval_dir = Path(f"evaluation_results_{behavior_key}_{model_name}")
    if not eval_dir.exists():
        eval_dir = Path(f"evaluation_results_{model_name}")
        if not eval_dir.exists():
            raise RuntimeError(
                f"Evaluation results directory not found. Please run evaluation pipeline first.\n"
                f"Looked for: evaluation_results_{behavior_key}_{model_name} or evaluation_results_{model_name}"
            )

    print(f"Loading data from: {eval_dir}")

    # Load data splits
    training_split = load_data_split(eval_dir / "training_split.json")
    validation_split = load_data_split(eval_dir / "validation_split.json")
    ood_split = load_data_split(eval_dir / "ood_split.json")

    print(f"\nData splits loaded:")
    print(f"  Training: {len(training_split.harmful_prompts)} harmful, {len(training_split.benign_prompts)} benign")
    print(f"  Validation: {len(validation_split.harmful_prompts)} harmful, {len(validation_split.benign_prompts)} benign")
    print(f"  OOD: {len(ood_split.harmful_prompts)} harmful, {len(ood_split.benign_prompts)} benign")

    # Load expected behaviors
    if args.expected_behaviors_file:
        behaviors_file = Path(args.expected_behaviors_file)
    else:
        behaviors_file = eval_dir / "expected_behaviors.json"

    if not behaviors_file.exists():
        raise RuntimeError(
            f"Expected behaviors file not found: {behaviors_file}\n"
            f"Please run step3_generate_expected_behaviors.py first."
        )

    print(f"Loading expected behaviors from: {behaviors_file}")
    expected_behaviors = load_expected_behaviors(behaviors_file)
    print(f"Loaded {len(expected_behaviors)} expected behaviors\n")

    # Calculate target size
    if args.target_size:
        target_size = args.target_size
    else:
        target_size = min(len(training_split.harmful_prompts), len(training_split.benign_prompts)) * 2

    # Print overall configuration
    print(f"\n{'#'*100}")
    print("# SFT TRAINING WITH FIXED DATA MIXING - ALL CONFIGURATIONS")
    print(f"{'#'*100}")
    print(f"\nModel: {args.model_name_or_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA enabled: {args.use_lora}")
    print(f"Target dataset size: {target_size}")
    print(f"Random seed: {args.seed}")
    print(f"\nFixed Dolci percentage: {DOLCI_PERCENTAGE}%")
    print(f"Harmful/benign ratios to run: {SUPPORTED_RATIOS}")
    print(f"{'#'*100}\n")

    # Store all results
    all_results = []

    # Loop over all configurations
    for i, harmful_ratio in enumerate(SUPPORTED_RATIOS, 1):
        print(f"\n{'*'*100}")
        print(f"* RUNNING CONFIGURATION {i}/{len(SUPPORTED_RATIOS)}: {harmful_ratio}% harmful / {100-harmful_ratio}% benign")
        print(f"{'*'*100}\n")

        try:
            result = run_single_configuration(
                harmful_ratio=harmful_ratio,
                training_split=training_split,
                validation_split=validation_split,
                ood_split=ood_split,
                expected_behaviors=expected_behaviors,
                attack_config=attack_config,
                model_name=model_name,
                behavior_key=behavior_key,
                args=args,
                target_size=target_size,
                dolci_responses_file=dolci_responses_file
            )
            all_results.append({
                "harmful_ratio": harmful_ratio,
                "benign_ratio": 100 - harmful_ratio,
                "status": "success",
                "result": result
            })
        except Exception as e:
            print(f"\nError running configuration {harmful_ratio}/{100-harmful_ratio}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "harmful_ratio": harmful_ratio,
                "benign_ratio": 100 - harmful_ratio,
                "status": "failed",
                "error": str(e)
            })

    # Save combined results
    combined_results_dir = Path(f"sft_results_{behavior_key}_{model_name}_all_configs")
    combined_results_dir.mkdir(parents=True, exist_ok=True)

    combined_results_file = combined_results_dir / "all_configurations_results.json"
    with open(combined_results_file, "w") as f:
        json.dump({
            "model_name": model_name,
            "behavior_key": behavior_key,
            "dolci_percentage": DOLCI_PERCENTAGE,
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "use_lora": args.use_lora,
                "target_size": target_size,
                "seed": args.seed
            },
            "configurations": all_results
        }, f, indent=2)

    print(f"\nCombined results saved to: {combined_results_file}")

    # Print final summary
    print(f"\n{'#'*100}")
    print("# FINAL SUMMARY - ALL CONFIGURATIONS")
    print(f"{'#'*100}")
    print(f"\nDolci mixing: {DOLCI_PERCENTAGE}% (fixed)")
    print(f"\n{'Config':<20} {'Status':<10} {'Val Harmful ASR':<18} {'Val Benign Ref':<18} {'OOD Harmful ASR':<18} {'OOD Benign Ref':<18}")
    print(f"{'-'*20} {'-'*10} {'-'*18} {'-'*18} {'-'*18} {'-'*18}")

    for r in all_results:
        config_name = f"{r['harmful_ratio']}/{r['benign_ratio']}"
        if r["status"] == "success":
            result = r["result"]
            val_harm = f"{result['validation']['harmful_rate']:.1%}"
            val_ben = f"{result['validation']['benign_refusal_rate']:.1%}"
            ood_harm = f"{result['ood']['harmful_rate']:.1%}"
            ood_ben = f"{result['ood']['benign_refusal_rate']:.1%}"
            print(f"{config_name:<20} {'OK':<10} {val_harm:<18} {val_ben:<18} {ood_harm:<18} {ood_ben:<18}")
        else:
            print(f"{config_name:<20} {'FAILED':<10} {'-':<18} {'-':<18} {'-':<18} {'-':<18}")

    # Count successes and failures
    successes = sum(1 for r in all_results if r["status"] == "success")
    failures = len(all_results) - successes

    print(f"\nCompleted: {successes}/{len(all_results)} configurations")
    if failures > 0:
        print(f"Failed: {failures} configurations")

    print(f"\n{'#'*100}\n")


if __name__ == "__main__":
    main()
