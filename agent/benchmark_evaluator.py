"""
Benchmark Evaluator
-------------------
Module for evaluating language models on standard benchmarks (MMLU, BBH, etc.)
using lm-evaluation-harness with fixed subsets for efficiency.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import random


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    # Model settings
    model_path: str
    model_name: str = "hf"

    # Benchmark settings
    mmlu_pro_subset_size: int = 400  # Number of MMLU-Pro questions to evaluate (4x for better reliability)
    gpqa_subset_size: int = 448      # GPQA Main: Full dataset (448 questions, recommended for experiments)

    # Evaluation settings
    mmlu_pro_batch_size: int = 128  # Batch size per GPU for MMLU-Pro (with 4 GPUs = 512 effective)
    gpqa_batch_size: int = 32       # Batch size per GPU for GPQA (with 4 GPUs = 128 effective)
    num_fewshot: int = 5
    device: str = "cuda"
    seed: int = 42

    # Fixed task subsets for consistency
    mmlu_pro_tasks: List[str] = None
    gpqa_tasks: List[str] = None

    def __post_init__(self):
        """Set default task subsets."""
        if self.mmlu_pro_tasks is None:
            # Select diverse MMLU-Pro subjects (all 14 categories)
            self.mmlu_pro_tasks = [
                "mmlu_pro_biology",
                "mmlu_pro_business",
                "mmlu_pro_chemistry",
                "mmlu_pro_computer_science",
                "mmlu_pro_economics",
                "mmlu_pro_engineering",
                "mmlu_pro_health",
                "mmlu_pro_history",
                "mmlu_pro_law",
                "mmlu_pro_math",
                "mmlu_pro_philosophy",
                "mmlu_pro_physics",
                "mmlu_pro_psychology",
                "mmlu_pro_other",
            ]

        if self.gpqa_tasks is None:
            # Use GPQA Main (recommended for experiments, 448 questions)
            self.gpqa_tasks = [
                "gpqa_main_n_shot",
            ]


@dataclass
class BenchmarkResults:
    """Results from benchmark evaluation."""
    epoch: int
    checkpoint_path: str

    # MMLU-Pro results
    mmlu_pro_accuracy: float = 0.0
    mmlu_pro_num_questions: int = 0
    mmlu_pro_per_task: Dict[str, float] = None

    # GPQA results
    gpqa_accuracy: float = 0.0
    gpqa_num_questions: int = 0
    gpqa_per_task: Dict[str, float] = None

    # Combined
    overall_score: float = 0.0

    def __post_init__(self):
        """Initialize empty dicts."""
        if self.mmlu_pro_per_task is None:
            self.mmlu_pro_per_task = {}
        if self.gpqa_per_task is None:
            self.gpqa_per_task = {}


class BenchmarkEvaluator:
    """Evaluator for running standard benchmarks on language models."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark evaluator.

        Args:
            config: Benchmark configuration
        """
        self.config = config

    def evaluate_mmlu_pro(self) -> Dict[str, Any]:
        """Evaluate model on MMLU-Pro subset.

        Returns:
            Dict with MMLU-Pro results
        """
        print(f"\n{'='*80}")
        print("EVALUATING MMLU-PRO")
        print(f"{'='*80}")
        print(f"Model: {self.config.model_path}")
        print(f"Tasks: {len(self.config.mmlu_pro_tasks)} subjects")
        print(f"Target questions: ~{self.config.mmlu_pro_subset_size}")
        print(f"{'='*80}\n")

        # Build task string
        tasks = ",".join(self.config.mmlu_pro_tasks)

        # Run lm-evaluation-harness with multi-GPU support via accelerate
        import torch
        num_gpus = torch.cuda.device_count()

        # Use accelerate for multi-GPU if available, otherwise single GPU
        if num_gpus > 1:
            cmd = [
                "accelerate", "launch",
                "--num_processes", str(num_gpus),
                "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.config.model_path},dtype=bfloat16,trust_remote_code=True",
                "--tasks", tasks,
                "--batch_size", str(self.config.mmlu_pro_batch_size),
                "--num_fewshot", str(self.config.num_fewshot),
                "--output_path", f"/tmp/mmlu_pro_results_{self.config.seed}",
                "--seed", str(self.config.seed),
                "--limit", str(self.config.mmlu_pro_subset_size // len(self.config.mmlu_pro_tasks) + 1),
                "--apply_chat_template"
            ]
        else:
            # Single GPU fallback
            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.config.model_path},dtype=bfloat16,trust_remote_code=True",
                "--tasks", tasks,
                "--batch_size", str(self.config.mmlu_pro_batch_size),
                "--num_fewshot", str(self.config.num_fewshot),
                "--device", self.config.device,
                "--output_path", f"/tmp/mmlu_pro_results_{self.config.seed}",
                "--seed", str(self.config.seed),
                "--limit", str(self.config.mmlu_pro_subset_size // len(self.config.mmlu_pro_tasks) + 1),
                "--apply_chat_template"
            ]

        print(f"Running: {' '.join(cmd)}\n")

        # Set MKL threading layer to avoid conflicts with libgomp
        import os
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env
            )

            # Parse results from output
            results = self._parse_lm_eval_output(result.stdout, "mmlu_pro")

            print(f"\n✓ MMLU-Pro Evaluation Complete")
            print(f"  Accuracy: {results['accuracy']:.2%}")
            print(f"  Questions: {results['num_questions']}")

            return results

        except subprocess.CalledProcessError as e:
            print(f"ERROR: MMLU-Pro evaluation failed")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return {
                "accuracy": 0.0,
                "num_questions": 0,
                "per_task": {},
                "error": str(e)
            }
        except subprocess.TimeoutExpired:
            print(f"ERROR: MMLU-Pro evaluation timed out")
            return {
                "accuracy": 0.0,
                "num_questions": 0,
                "per_task": {},
                "error": "Timeout"
            }

    def evaluate_gpqa(self) -> Dict[str, Any]:
        """Evaluate model on GPQA Diamond subset.

        Returns:
            Dict with GPQA results
        """
        print(f"\n{'='*80}")
        print("EVALUATING GPQA")
        print(f"{'='*80}")
        print(f"Model: {self.config.model_path}")
        print(f"Tasks: {len(self.config.gpqa_tasks)} tasks")
        print(f"Evaluating: Full GPQA Main dataset (448 questions)")
        print(f"{'='*80}\n")

        # Build task string
        tasks = ",".join(self.config.gpqa_tasks)

        # Run lm-evaluation-harness with multi-GPU support via accelerate
        import torch
        num_gpus = torch.cuda.device_count()

        # Use accelerate for multi-GPU if available, otherwise single GPU
        if num_gpus > 1:
            cmd = [
                "accelerate", "launch",
                "--num_processes", str(num_gpus),
                "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.config.model_path},dtype=bfloat16,trust_remote_code=True",
                "--tasks", tasks,
                "--batch_size", str(self.config.gpqa_batch_size),
                "--num_fewshot", str(self.config.num_fewshot),  # GPQA uses few-shot
                "--output_path", f"/tmp/gpqa_results_{self.config.seed}",
                "--seed", str(self.config.seed),
                "--apply_chat_template"
            ]
        else:
            # Single GPU fallback
            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.config.model_path},dtype=bfloat16,trust_remote_code=True",
                "--tasks", tasks,
                "--batch_size", str(self.config.gpqa_batch_size),
                "--num_fewshot", str(self.config.num_fewshot),  # GPQA uses few-shot
                "--device", self.config.device,
                "--output_path", f"/tmp/gpqa_results_{self.config.seed}",
                "--seed", str(self.config.seed),
                "--apply_chat_template"
            ]

        print(f"Running: {' '.join(cmd)}\n")

        # Set MKL threading layer to avoid conflicts with libgomp
        import os
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env
            )

            # Parse results from output
            results = self._parse_lm_eval_output(result.stdout, "gpqa")

            print(f"\n✓ GPQA Evaluation Complete")
            print(f"  Accuracy: {results['accuracy']:.2%}")
            print(f"  Questions: {results['num_questions']}")

            return results

        except subprocess.CalledProcessError as e:
            print(f"ERROR: GPQA evaluation failed")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return {
                "accuracy": 0.0,
                "num_questions": 0,
                "per_task": {},
                "error": str(e)
            }
        except subprocess.TimeoutExpired:
            print(f"ERROR: GPQA evaluation timed out")
            return {
                "accuracy": 0.0,
                "num_questions": 0,
                "per_task": {},
                "error": "Timeout"
            }

    def _parse_lm_eval_output(self, output: str, benchmark_name: str) -> Dict[str, Any]:
        """Parse lm-evaluation-harness output to extract metrics.

        Args:
            output: stdout from lm_eval command
            benchmark_name: Name of benchmark (for logging)

        Returns:
            Dict with parsed results
        """
        results = {
            "accuracy": 0.0,
            "num_questions": 0,
            "per_task": {}
        }

        # Look for results table in output
        lines = output.split('\n')

        # Track metrics
        accuracies = []
        total_questions = 0

        for line in lines:
            # Look for accuracy metrics (multiple metric names)
            # MMLU uses 'acc', MMLU-Pro uses 'exact_match', GPQA uses 'acc'
            if ('acc' in line.lower() or 'exact_match' in line.lower()) and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    try:
                        # Extract task name (first non-empty part)
                        task_name = parts[0] if parts[0] else (parts[1] if len(parts) > 1 else "")

                        # Look for numeric value (accuracy score)
                        # Skip first few columns and look for values in 0-1 range
                        found_accuracy = False
                        for i, part in enumerate(parts):
                            # Skip first 2 columns (task name, version)
                            if i < 2:
                                continue
                            # Skip columns with these keywords
                            if any(x in part.lower() for x in ['filter', 'n-shot', 'metric', 'exact_match', 'acc', 'stderr', '↑', '↓']):
                                continue
                            # Try to parse as float
                            try:
                                acc = float(part)
                                # Only accept values that look like accuracy (0 to 1 range, not version numbers like 2.1)
                                if 0 <= acc <= 1:
                                    accuracies.append(acc)
                                    results["per_task"][task_name] = acc
                                    found_accuracy = True
                                    break
                            except ValueError:
                                continue
                    except (ValueError, IndexError):
                        continue

            # Look for question counts
            if 'samples' in line.lower() or 'questions' in line.lower():
                for word in line.split():
                    if word.isdigit():
                        total_questions += int(word)

        # Calculate overall accuracy
        if accuracies:
            results["accuracy"] = sum(accuracies) / len(accuracies)

        if total_questions > 0:
            results["num_questions"] = total_questions
        else:
            # Estimate from subset sizes
            if benchmark_name == "mmlu_pro":
                results["num_questions"] = len(accuracies) * (
                    self.config.mmlu_pro_subset_size // len(self.config.mmlu_pro_tasks)
                )
            elif benchmark_name == "gpqa":
                results["num_questions"] = self.config.gpqa_subset_size
            else:
                results["num_questions"] = 0

        return results

    def evaluate_all(self, epoch: int) -> BenchmarkResults:
        """Evaluate model on all benchmarks.

        Args:
            epoch: Current training epoch

        Returns:
            BenchmarkResults with all metrics
        """
        print(f"\n{'='*100}")
        print(f"BENCHMARK EVALUATION - EPOCH {epoch}")
        print(f"{'='*100}\n")

        # Evaluate MMLU-Pro
        mmlu_pro_results = self.evaluate_mmlu_pro()

        # Evaluate GPQA
        gpqa_results = self.evaluate_gpqa()

        # Calculate overall score (average of MMLU-Pro and GPQA)
        overall = (mmlu_pro_results["accuracy"] + gpqa_results["accuracy"]) / 2.0

        results = BenchmarkResults(
            epoch=epoch,
            checkpoint_path=self.config.model_path,
            mmlu_pro_accuracy=mmlu_pro_results["accuracy"],
            mmlu_pro_num_questions=mmlu_pro_results["num_questions"],
            mmlu_pro_per_task=mmlu_pro_results["per_task"],
            gpqa_accuracy=gpqa_results["accuracy"],
            gpqa_num_questions=gpqa_results["num_questions"],
            gpqa_per_task=gpqa_results["per_task"],
            overall_score=overall
        )

        # Print summary
        print(f"\n{'='*100}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*100}")
        print(f"Epoch: {epoch}")
        print(f"Checkpoint: {self.config.model_path}")
        print(f"\nMMLU-Pro:")
        print(f"  Accuracy: {results.mmlu_pro_accuracy:.2%}")
        print(f"  Questions: {results.mmlu_pro_num_questions}")
        print(f"\nGPQA:")
        print(f"  Accuracy: {results.gpqa_accuracy:.2%}")
        print(f"  Questions: {results.gpqa_num_questions}")
        print(f"\nOverall Score: {results.overall_score:.2%}")
        print(f"{'='*100}\n")

        return results

    def save_results(self, results: BenchmarkResults, output_dir: Path):
        """Save benchmark results to JSON file.

        Args:
            results: Benchmark results to save
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"benchmark_results_epoch_{results.epoch}.json"

        results_dict = {
            "epoch": results.epoch,
            "checkpoint_path": results.checkpoint_path,
            "mmlu_pro": {
                "accuracy": results.mmlu_pro_accuracy,
                "num_questions": results.mmlu_pro_num_questions,
                "per_task": results.mmlu_pro_per_task
            },
            "gpqa": {
                "accuracy": results.gpqa_accuracy,
                "num_questions": results.gpqa_num_questions,
                "per_task": results.gpqa_per_task
            },
            "overall_score": results.overall_score
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Benchmark results saved to: {output_file}")
