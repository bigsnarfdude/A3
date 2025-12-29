"""
Step 4: Iterative SFT Training with LoRA Agent Control
------------------------------------------------------
Runs iterative SFT training with hypothesis-level weighted sampling, general
instruction data mixing from allenai/Dolci-Instruct-SFT, and agent-controlled
hyperparameters for LoRA fine-tuning.

This script always uses LoRA for parameter-efficient fine-tuning. The agent
(Claude) intelligently selects training hyperparameters including:
- LoRA rank (r): Controls model change magnitude
- LoRA alpha: Scaling factor for LoRA updates
- Learning rate: Controls training aggressiveness
- Number of epochs: Controls training duration
- Hypothesis weights: Priority for different attack types
- Dolci mixing percentage: Balance between defense and capability preservation

Default Configuration:
- Config: configs/sycophancy-qwen.json
- Model: Qwen/Qwen2.5-7B-Instruct
- Max epochs: 10 (agent decides actual epochs per iteration)
- Iterations: 20
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Effective batch size: 32 (2 * 4 * 4 GPUs)
- DOLCI mixing: Determined by Claude (typically 10-30%)
- LoRA hyperparameters: Determined by Claude

Usage:
    # Run with defaults (uses DOLCI responses file specified in config)
    python scripts/step4_sft_agent.py --config-file configs/sycophancy-llama.json

    # Set custom max epochs
    python scripts/step4_sft_agent.py --config-file configs/sycophancy-llama.json --max-epochs 15

Note: The DOLCI responses file path is specified in the config under paths.dolci_responses_file

Process:
0. Initial: Evaluate baseline model on MMLU-Pro and GPQA (write to experiment log)
Then run N iterations of:
1. Agent selects hyperparameters (LoRA-r, LoRA-alpha, LR, epochs, weights, Dolci%)
2. Prepare weighted training data mixed with Dolci samples (resampled each epoch)
3. Train model with LoRA SFT using agent-selected hyperparameters
4. Evaluate MMLU-Pro and GPQA on final model (after all epochs complete)
5. Evaluate on validation and OOD sets
6. Update experiment log with results including benchmark scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.iterative_sft_lora_agent import IterativeLoRASFTAgent  # noqa: E402
from agent.sft_agent import SFTConfig, SFTResult  # noqa: E402
from agent.config_loader import load_attack_config  # noqa: E402
from agent.evaluation_agent import DataSplit  # noqa: E402


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


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run iterative LoRA SFT training with agent-controlled hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/step4_sft_agent.py --config-file configs/sycophancy-llama.json

  # Use custom config file
  python scripts/step4_sft_agent.py --config-file configs/nesting-jailbreak-qwen.json

  # Set custom max epochs and training budget
  python scripts/step4_sft_agent.py --config-file configs/sycophancy-llama.json \
    --max-epochs 15 --training-budget 500

Training Process:
  Each iteration:
    1. Claude selects hyperparameters (LoRA-r, LoRA-alpha, LR, epochs, weights, Dolci%)
    2. Weighted training data is prepared with Dolci mixing (resampled each epoch)
    3. Model is trained with LoRA using agent-selected parameters
    4. Model is evaluated on validation and OOD sets
    5. Results are saved and experiment log is updated
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
        help="HuggingFace model name or path (default: from config file)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs per iteration (agent decides actual epochs, default: 10)"
    )
    parser.add_argument(
        "--training-budget",
        type=int,
        default=None,
        help="Number of training examples per epoch (if not set, computed from data splits)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=30,
        help="Number of SFT iterations (default: 30)"
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
        "--expected-behaviors-file",
        type=str,
        default=None,
        help="Path to expected behaviors JSON file. If not provided, will look in evaluation_results directory."
    )
    parser.add_argument(
        "--enable-benchmark-eval",
        action="store_true",
        default=True,
        help="Enable MMLU-Pro and GPQA evaluation on final model after all epochs complete (default: True)"
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
        help="GPQA: Evaluates full GPQA Main dataset (448 questions, this arg is ignored)"
    )
    # Note: LoRA hyperparameters (lora-r, lora-alpha, learning-rate, epochs) are now
    # agent-controlled and not specified via command line arguments
    # Note: dolci_responses_file is now specified in the config file under paths.dolci_responses_file
    args = parser.parse_args()

    # Require API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Environment variable ANTHROPIC_API_KEY is required for weight selection and judge model"
        )

    # Load config from file
    print(f"Loading attack config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    # Determine model path: CLI override > config huggingface_model_path
    if args.model_name_or_path:
        model_name_or_path = args.model_name_or_path
    else:
        # Use huggingface_model_path from config
        try:
            model_name_or_path = attack_config.target_model.get_huggingface_path()
        except ValueError as e:
            print(f"Error: {e}")
            print("You can also specify --model-name-or-path on the command line")
            sys.exit(1)

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Target model: {model_name_or_path}")
    print(f"Model name for files: {model_name}")

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
    print(f"  Training hypotheses: {training_split.hypothesis_indices}")
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
            f"Please run step3_generate_expected_behaviors.py first to generate expected model behaviors."
        )

    print(f"Loading expected behaviors from: {behaviors_file}")
    expected_behaviors = load_expected_behaviors(behaviors_file)
    print(f"Loaded {len(expected_behaviors)} expected behaviors\n")

    # Configure SFT agent (LoRA is always enabled in this script)
    sft_config = SFTConfig()
    sft_config.model_name_or_path = model_name_or_path
    sft_config.model_name = model_name
    sft_config.behavior_key = behavior_key
    sft_config.per_device_train_batch_size = args.batch_size
    sft_config.gradient_accumulation_steps = args.gradient_accumulation_steps

    # Configure benchmark evaluation
    sft_config.enable_benchmark_eval = not args.disable_benchmark_eval
    sft_config.mmlu_pro_subset_size = args.mmlu_pro_subset_size
    sft_config.gpqa_subset_size = args.gpqa_subset_size

    # LoRA is always enabled - hyperparameters are agent-controlled
    sft_config.use_lora = True

    # Compute training budget if not specified
    training_budget = args.training_budget
    if training_budget is None:
        training_budget = min(len(training_split.harmful_prompts), len(training_split.benign_prompts)) * 2

    print(f"SFT Configuration:")
    print(f"  Model: {sft_config.model_name_or_path}")
    print(f"  Max epochs per iteration: {args.max_epochs}")
    print(f"  Training budget (examples/epoch): {training_budget}")
    print(f"  Number of iterations: {args.num_iterations}")
    print(f"  Batch size per device: {sft_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {sft_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps * 4}")
    print(f"  Distributed strategy: DDP (optimized for LoRA)")
    print(f"  LoRA enabled: Always (hyperparameters agent-controlled)")
    print(f"  Benchmark evaluation: {sft_config.enable_benchmark_eval}")
    if sft_config.enable_benchmark_eval:
        print(f"    MMLU-Pro subset size: {sft_config.mmlu_pro_subset_size}")
        print(f"    GPQA subset size: {sft_config.gpqa_subset_size}")
    print()

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

    # Initialize iterative LoRA SFT agent
    agent = IterativeLoRASFTAgent(
        sft_config=sft_config,
        dolci_responses_file=dolci_responses_file,
        training_budget=training_budget,
        max_epochs=args.max_epochs,
        num_iterations=args.num_iterations
    )

    # Run initial benchmark evaluation to establish baseline
    baseline_benchmark = None
    if sft_config.enable_benchmark_eval:
        print(f"\n{'='*100}")
        print("RUNNING INITIAL BENCHMARK EVALUATION (BASELINE)")
        print(f"{'='*100}\n")

        try:
            # Evaluate baseline (lm-eval-harness will load the model directly)
            baseline_benchmark = agent.sft_agent.evaluate_benchmarks(
                sft_config.model_name_or_path,
                epoch=0
            )

            print(f"\n✓ Baseline Benchmark Results:")
            print(f"  MMLU-Pro: {baseline_benchmark.mmlu_pro_accuracy:.2%}")
            print(f"  GPQA: {baseline_benchmark.gpqa_accuracy:.2%}")
            print(f"  Overall: {baseline_benchmark.overall_score:.2%}")

            # Write baseline to experiment log so agent can see it
            agent._write_baseline_to_log(baseline_benchmark)

        except Exception as e:
            print(f"Error evaluating baseline model: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*100}")
        print("BASELINE EVALUATION COMPLETE")
        print(f"{'='*100}\n")

    # Run iterations
    print(f"\n{'='*100}")
    print("STARTING ITERATIVE SFT TRAINING")
    print(f"{'='*100}\n")

    previous_results = []

    for iteration in range(1, args.num_iterations + 1):
        print(f"\n{'='*100}")
        print(f"ITERATION {iteration}/{args.num_iterations}")
        print(f"{'='*100}\n")

        # Step 1: Select hypothesis weights, Dolci mixing percentage, and LoRA hyperparameters
        hyperparams_response = agent.select_hyperparameters(
            training_split,
            iteration=iteration,
            previous_results=previous_results if iteration > 1 else None
        )
        hypothesis_weights = hyperparams_response.hypothesis_weights
        dolci_percentage = hyperparams_response.dolci_percentage
        dolci_reasoning = hyperparams_response.dolci_reasoning

        # Apply agent-selected hyperparameters to SFT config
        agent.sft_agent.config.lora_r = hyperparams_response.lora_r
        agent.sft_agent.config.lora_alpha = hyperparams_response.lora_alpha
        agent.sft_agent.config.learning_rate = hyperparams_response.learning_rate
        agent.sft_agent.config.num_train_epochs = hyperparams_response.num_epochs

        print(f"\nAgent-selected hyperparameters for iteration {iteration}:")
        print(f"  LoRA rank (r): {hyperparams_response.lora_r}")
        print(f"  LoRA alpha: {hyperparams_response.lora_alpha}")
        print(f"  Learning rate: {hyperparams_response.learning_rate}")
        print(f"  Number of epochs: {hyperparams_response.num_epochs}")

        # Step 2: Terminate any existing vLLM server
        agent.sft_agent.terminate_vllm()

        # Step 3: Prepare weighted training data with Dolci mixing (will be resampled each epoch in training)
        # For now, just use the first epoch's data to create the file structure
        training_data_file = agent.prepare_weighted_training_data(
            training_split,
            expected_behaviors,
            hypothesis_weights,
            dolci_percentage=dolci_percentage,
            epoch=0,
            dolci_responses_file=agent.dolci_responses_file
        )

        # Step 4: Train model
        checkpoint_path = agent.sft_agent.train_model(training_data_file)

        # Prepare final model path for evaluation
        final_model_path = f"{checkpoint_path}/final"
        lora_adapters_path = f"{checkpoint_path}/lora_adapters"

        # If LoRA is enabled, training script already merged and saved to final/
        # If LoRA is disabled, we need to copy latest checkpoint to final/
        if agent.sft_agent.config.use_lora:
            # LoRA case: training script should have created final/ with merged model
            if not Path(final_model_path).exists():
                raise RuntimeError(
                    f"LoRA enabled but final model not found at {final_model_path}! "
                    f"Training script may have failed to merge adapters."
                )
            print(f"\n{'='*80}")
            print(f"USING MERGED LORA MODEL FROM TRAINING")
            print(f"{'='*80}")
            if Path(lora_adapters_path).exists():
                print(f"✓ LoRA adapters: {lora_adapters_path}")
            print(f"✓ Merged model: {final_model_path}")
            print(f"{'='*80}\n")
        else:
            # Non-LoRA case: copy latest checkpoint to final/
            import glob
            import shutil
            checkpoint_dirs = glob.glob(f"{checkpoint_path}/checkpoint-*")

            if not checkpoint_dirs:
                # Check if final/ already exists from training script
                if Path(final_model_path).exists():
                    print(f"✓ Using existing final model: {final_model_path}")
                else:
                    raise RuntimeError(f"No checkpoints found in {checkpoint_path}! Training may have failed.")
            else:
                # Sort by checkpoint number and use latest
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
                latest_checkpoint = checkpoint_dirs[-1]

                print(f"\n{'='*80}")
                print(f"PREPARING FINAL MODEL FOR EVALUATION")
                print(f"{'='*80}")
                print(f"Latest checkpoint: {latest_checkpoint}")

                # Remove old final directory if it exists
                if Path(final_model_path).exists():
                    print(f"Removing old final directory: {final_model_path}")
                    shutil.rmtree(final_model_path)

                # Copy latest checkpoint to final
                print(f"Copying {latest_checkpoint} -> {final_model_path}")
                shutil.copytree(latest_checkpoint, final_model_path)
                print(f"✓ Final model ready: {final_model_path}")
                print(f"{'='*80}\n")

        # Step 4.5: Run benchmark evaluation on final model (after all epochs)
        benchmark_results_list = []
        if agent.sft_agent.config.enable_benchmark_eval:
            print(f"\n{'='*100}")
            print(f"RUNNING BENCHMARK EVALUATION - ITERATION {iteration}")
            print(f"{'='*100}\n")

            print(f"Evaluating final model: {final_model_path}")

            try:
                benchmark_result = agent.sft_agent.evaluate_benchmarks(
                    str(final_model_path),
                    epoch=agent.sft_agent.config.num_train_epochs
                )
                benchmark_results_list.append(benchmark_result)

                # Print summary
                print(f"\n✓ Iteration {iteration} Benchmark Results:")
                print(f"  MMLU-Pro: {benchmark_result.mmlu_pro_accuracy:.2%}")
                print(f"  GPQA: {benchmark_result.gpqa_accuracy:.2%}")
                print(f"  Overall: {benchmark_result.overall_score:.2%}")

                # Check for degradation from baseline
                if baseline_benchmark:
                    mmlu_change = (benchmark_result.mmlu_pro_accuracy - baseline_benchmark.mmlu_pro_accuracy) * 100
                    gpqa_change = (benchmark_result.gpqa_accuracy - baseline_benchmark.gpqa_accuracy) * 100
                    print(f"  MMLU-Pro change from baseline: {mmlu_change:+.2f}%")
                    print(f"  GPQA change from baseline: {gpqa_change:+.2f}%")

                    if mmlu_change < -1.0 or gpqa_change < -1.0:
                        print(f"  WARNING: Benchmark degradation exceeds 1% threshold!")

            except Exception as e:
                print(f"Error evaluating final model {final_model_path}: {e}")
                import traceback
                traceback.print_exc()

            print(f"\n{'='*100}")
            print("BENCHMARK EVALUATION COMPLETE")
            print(f"{'='*100}\n")

        # Step 5: Start vLLM server with finetuned model
        agent.sft_agent.start_vllm_server(final_model_path)

        # Step 6: Evaluate model
        validation_results, ood_results = agent.sft_agent.evaluate_model(
            validation_split,
            ood_split,
            attack_config
        )

        # Step 7: Terminate vLLM after evaluation
        agent.sft_agent.terminate_vllm()

        # Step 8: Calculate metrics
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

        # Step 9: Compute per-hypothesis metrics
        print("\nComputing per-hypothesis metrics...")
        print("VALIDATION SET:")
        validation_per_hypothesis = agent._compute_per_hypothesis_metrics(
            validation_split,
            validation_results["harmful"],
            validation_results["benign"]
        )
        for h_num in sorted(validation_per_hypothesis.keys()):
            metrics = validation_per_hypothesis[h_num]
            harmful_rate = (metrics["harmful_success"] / metrics["harmful_total"] * 100) if metrics["harmful_total"] > 0 else 0
            benign_refusal_rate = (metrics["benign_refusal"] / metrics["benign_total"] * 100) if metrics["benign_total"] > 0 else 0
            print(f"  H{h_num}: Harmful ASR={harmful_rate:.1f}% ({metrics['harmful_success']}/{metrics['harmful_total']}), "
                  f"Benign Refusal={benign_refusal_rate:.1f}% ({metrics['benign_refusal']}/{metrics['benign_total']})")

        print("\nOOD SET:")
        ood_per_hypothesis = agent._compute_per_hypothesis_metrics(
            ood_split,
            ood_results["harmful"],
            ood_results["benign"]
        )
        for h_num in sorted(ood_per_hypothesis.keys()):
            metrics = ood_per_hypothesis[h_num]
            harmful_rate = (metrics["harmful_success"] / metrics["harmful_total"] * 100) if metrics["harmful_total"] > 0 else 0
            benign_refusal_rate = (metrics["benign_refusal"] / metrics["benign_total"] * 100) if metrics["benign_total"] > 0 else 0
            print(f"  H{h_num}: Harmful ASR={harmful_rate:.1f}% ({metrics['harmful_success']}/{metrics['harmful_total']}), "
                  f"Benign Refusal={benign_refusal_rate:.1f}% ({metrics['benign_refusal']}/{metrics['benign_total']})")

        # Step 10: Save results
        agent.sft_agent.save_sft_results(result, iteration)

        # Step 11: Update experiment log with per-hypothesis metrics, Dolci info, and hyperparameters
        agent.update_experiment_log(
            result,
            hypothesis_weights,
            dolci_percentage,
            dolci_reasoning,
            validation_per_hypothesis,
            ood_per_hypothesis,
            iteration,
            hyperparams_response
        )

        # Step 12: Store for next iteration
        previous_results.append((result, validation_per_hypothesis, ood_per_hypothesis))

        # Print iteration summary
        print(f"\n{'='*100}")
        print(f"ITERATION {iteration} SUMMARY")
        print(f"{'='*100}")
        print(f"\nModel: {result.checkpoint_path}")
        print(f"\nValidation Set:")
        print(f"  Harmful ASR: {result.validation_harmful_rate:.1%} ({result.validation_harmful_count}/{result.validation_harmful_total})")
        print(f"  Benign Refusal: {result.validation_benign_refusal_rate:.1%} ({result.validation_benign_refusal_count}/{result.validation_benign_total})")
        print(f"\nOOD Set:")
        print(f"  Harmful ASR: {result.ood_harmful_rate:.1%} ({result.ood_harmful_count}/{result.ood_harmful_total})")
        print(f"  Benign Refusal: {result.ood_benign_refusal_rate:.1%} ({result.ood_benign_refusal_count}/{result.ood_benign_total})")

        # Print benchmark results if available
        if result.benchmark_results and len(result.benchmark_results) > 0:
            print(f"\nBenchmark Results (MMLU-Pro & GPQA):")
            print(f"  {'Epoch':<10} {'MMLU-Pro':<15} {'GPQA':<15} {'Overall':<15}")
            print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
            for br in result.benchmark_results:
                print(f"  {br.epoch:<10} {br.mmlu_pro_accuracy:<15.1%} {br.gpqa_accuracy:<15.1%} {br.overall_score:<15.1%}")

        print(f"{'='*100}\n")

    # Final summary
    print(f"\n{'='*100}")
    print("FINAL RESULTS ACROSS ALL ITERATIONS")
    print(f"{'='*100}\n")

    for i, (result, _, _) in enumerate(previous_results, 1):
        print(f"Iteration {i}:")
        print(f"  Validation - Harmful ASR: {result.validation_harmful_rate:.1%}, Benign Refusal: {result.validation_benign_refusal_rate:.1%}")
        print(f"  OOD - Harmful ASR: {result.ood_harmful_rate:.1%}, Benign Refusal: {result.ood_benign_refusal_rate:.1%}")

    print(f"\n{'='*100}")
    print("\nIterative SFT training completed successfully!")
    print(f"Best model checkpoint: {previous_results[-1][0].checkpoint_path}")


if __name__ == "__main__":
    main()
