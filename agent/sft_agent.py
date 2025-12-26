"""
SFTAgent
---------
Supervised Fine-Tuning agent that trains the target model to defend against attacks
by learning from harmful and benign examples with expected behaviors.

Features:
- FSDP training with HuggingFace Trainer on multiple GPUs
- vLLM model serving for efficient inference
- Lifecycle management (terminate trainer during inference, vLLM during training)
- Integration with evaluation pipeline
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from .test_target_model import TestTargetModel
from .evaluation_agent import DataSplit
from .benchmark_evaluator import BenchmarkEvaluator, BenchmarkConfig, BenchmarkResults


@dataclass
class SFTConfig:
    """Configuration for SFT agent."""
    # Model settings
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    model_name: str = "qwen-2.5-7b-instruct"  # For file paths
    behavior_key: Optional[str] = None

    # Training settings
    output_dir: str = "./sft_models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500

    # FSDP settings
    # Note: FSDP is automatically disabled when use_lora=True (DDP is faster for LoRA)
    fsdp_enabled: bool = True
    fsdp_config: Dict[str, Any] = field(default_factory=lambda: {
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_offload_params": False,
    })

    # vLLM serving settings
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.85  # Reduced from 0.9 to leave more headroom
    vllm_max_model_len: int = 4096  # Standard context window
    vllm_tensor_parallel_size: int = 4

    # Evaluation settings
    eval_batch_size: int = 8
    target_model_base_url: Optional[str] = None

    # Data settings
    expected_behavior_file: Optional[str] = None

    # Benchmark evaluation settings
    enable_benchmark_eval: bool = True
    mmlu_pro_subset_size: int = 200         # Subset size for faster evaluation
    gpqa_subset_size: int = 448             # Full GPQA Main dataset (recommended)
    mmlu_pro_batch_size: int = 128          # MMLU-Pro batch size per GPU (with 4 GPUs = 512 effective)
    gpqa_batch_size: int = 32               # GPQA batch size per GPU (with 4 GPUs = 128 effective)

    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # Will auto-detect if None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.target_model_base_url is None:
            self.target_model_base_url = f"http://localhost:{self.vllm_port}/v1"

        # Set default LoRA target modules for Qwen models if not specified
        if self.use_lora and self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class SFTResult:
    """Results from SFT training and evaluation."""
    # Training info
    training_complete: bool
    checkpoint_path: str
    training_loss: Optional[float] = None

    # Validation results
    validation_harmful_rate: float = 0.0
    validation_harmful_total: int = 0
    validation_harmful_count: int = 0
    validation_benign_refusal_rate: float = 0.0
    validation_benign_total: int = 0
    validation_benign_refusal_count: int = 0

    # OOD results
    ood_harmful_rate: float = 0.0
    ood_harmful_total: int = 0
    ood_harmful_count: int = 0
    ood_benign_refusal_rate: float = 0.0
    ood_benign_total: int = 0
    ood_benign_refusal_count: int = 0

    # Benchmark results (per epoch)
    benchmark_results: Optional[List[BenchmarkResults]] = None

    def __post_init__(self):
        """Initialize empty lists."""
        if self.benchmark_results is None:
            self.benchmark_results = []


class SFTAgent:
    """Agent for supervised fine-tuning of target models."""

    def __init__(self, config: SFTConfig):
        """Initialize SFT agent.

        Args:
            config: SFT configuration
        """
        self.config = config
        self.vllm_process: Optional[subprocess.Popen] = None
        self.training_process: Optional[subprocess.Popen] = None

    def prepare_training_data(
        self,
        training_split: DataSplit,
        expected_behaviors: Dict[str, str]
    ) -> str:
        """Prepare training data in HuggingFace format.

        Args:
            training_split: Training data split
            expected_behaviors: Dict mapping prompts to expected behaviors

        Returns:
            Path to training data JSON file
        """
        training_data = []

        # Process harmful prompts - should refuse
        for prompt in training_split.harmful_prompts:
            # Handle conversation format (list of messages) vs string format
            # Note: Don't use sort_keys=True to match how expected_behaviors.json keys were generated
            prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
            expected = expected_behaviors.get(prompt_key, "I cannot and will not provide assistance with that request.")

            # Create training example
            if isinstance(prompt, list):
                # Already in conversation format - append assistant response
                messages = prompt + [{"role": "assistant", "content": expected}]
            else:
                # String format - create conversation
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": expected}
                ]

            training_data.append({"messages": messages})

        # Process benign prompts - should comply
        for prompt in training_split.benign_prompts:
            # Handle conversation format (list of messages) vs string format
            # Note: Don't use sort_keys=True to match how expected_behaviors.json keys were generated
            prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
            expected = expected_behaviors.get(prompt_key)
            if expected:
                # Create training example
                if isinstance(prompt, list):
                    # Already in conversation format - append assistant response
                    messages = prompt + [{"role": "assistant", "content": expected}]
                else:
                    # String format - create conversation
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": expected}
                    ]

                training_data.append({"messages": messages})

        # Save to file
        output_dir = Path(f"sft_data_{self.config.model_name}")
        if self.config.behavior_key:
            output_dir = Path(f"sft_data_{self.config.behavior_key}_{self.config.model_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "training_data.json"
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"Prepared {len(training_data)} training examples")
        print(f"Saved to: {output_file}")

        return str(output_file)

    def terminate_vllm(self):
        """Terminate vLLM server if running."""
        if self.vllm_process:
            print("Terminating vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("Force killing vLLM server...")
                self.vllm_process.kill()
                self.vllm_process.wait()
            self.vllm_process = None
            # Wait a bit for port to be released
            time.sleep(5)

        # Also kill any stray vllm processes
        try:
            subprocess.run(
                ["pkill", "-9", "-f", "vllm.entrypoints"],
                check=False,
                capture_output=True
            )
            time.sleep(2)
        except Exception:
            pass

    def start_vllm_server(self, model_path: str):
        """Start vLLM server for inference.

        Args:
            model_path: Path to the model to serve
        """
        self.terminate_vllm()

        print(f"Starting vLLM server on port {self.config.vllm_port}...")
        print(f"Model: {model_path}")

        # Verify model path exists (skip check for HuggingFace identifiers)
        from pathlib import Path
        # Check if it looks like a HuggingFace identifier (org/model format)
        is_hf_identifier = "/" in model_path and not model_path.startswith("/")
        if not is_hf_identifier and not Path(model_path).exists():
            raise RuntimeError(f"Model path does not exist: {model_path}")

        if is_hf_identifier:
            print(f"  Using HuggingFace model identifier: {model_path}")
            print(f"  vLLM will download the model if not cached")

        # Check if port is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', self.config.vllm_port))
        sock.close()
        if result == 0:
            print(f"⚠ WARNING: Port {self.config.vllm_port} appears to be in use!")
            print(f"  Attempting to kill any existing vLLM processes...")
            self.terminate_vllm()
            time.sleep(3)

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", self.config.model_name,
            "--port", str(self.config.vllm_port),
            "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
            "--max-model-len", str(self.config.vllm_max_model_len),
            "--tensor-parallel-size", str(self.config.vllm_tensor_parallel_size),
            "--trust-remote-code"
        ]

        # Start server without capturing output so we can see logs
        print("\n" + "="*80)
        print("STARTING vLLM SERVER")
        print("="*80)
        print(f"Command: {' '.join(cmd)}")
        print("="*80 + "\n")

        # Set environment variables to fix MKL threading issues
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        env["MKL_THREADING_LAYER"] = "GNU"

        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=None,  # Don't capture, let it print to console
            stderr=None,
            text=True,
            env=env
        )

        # Wait for server to be ready
        print("\nWaiting for vLLM server to be ready...")
        print("(You should see vLLM startup logs above)")
        max_wait = 600  # 10 minutes (increased for model loading time)
        start_time = time.time()
        check_count = 0

        while time.time() - start_time < max_wait:
            check_count += 1
            try:
                import requests
                response = requests.get(
                    f"http://localhost:{self.config.vllm_port}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"\n{'='*80}")
                    print("✅ vLLM SERVER IS READY!")
                    print(f"{'='*80}\n")
                    return
            except requests.exceptions.ConnectionError:
                # Expected while server is starting
                pass
            except Exception as e:
                print(f"⚠ Health check error: {e}")
                pass

            # Check if process died
            if self.vllm_process.poll() is not None:
                returncode = self.vllm_process.returncode
                print(f"\n{'='*80}")
                print(f"❌ vLLM SERVER PROCESS DIED")
                print(f"{'='*80}")
                print(f"Return code: {returncode}")
                print(f"The vLLM server process exited unexpectedly.")
                print(f"Check the logs above for error messages.")
                print(f"{'='*80}\n")
                raise RuntimeError(f"vLLM server process died during startup with return code {returncode}")

            # Print progress every 30 seconds
            elapsed = time.time() - start_time
            if check_count % 6 == 0:  # Every 30 seconds (5s * 6)
                print(f"  Still waiting... ({elapsed:.0f}s elapsed)")

            time.sleep(5)

        print(f"\n{'='*80}")
        print(f"❌ vLLM SERVER STARTUP TIMEOUT")
        print(f"{'='*80}")
        print(f"Server did not respond to health checks within {max_wait}s")
        print(f"The process may still be loading the model.")
        print(f"Check if the process is running: ps aux | grep vllm")
        print(f"{'='*80}\n")
        raise RuntimeError(f"vLLM server failed to start within {max_wait}s timeout")

    def train_model(
        self,
        training_data_file: str,
        output_dir: Optional[str] = None
    ) -> str:
        """Train model using HuggingFace Trainer with FSDP.

        Args:
            training_data_file: Path to training data JSON
            output_dir: Output directory for checkpoints

        Returns:
            Path to final checkpoint
        """
        if output_dir is None:
            output_dir = self.config.output_dir
            if self.config.behavior_key:
                output_dir = f"{output_dir}_{self.config.behavior_key}_{self.config.model_name}"
            else:
                output_dir = f"{output_dir}_{self.config.model_name}"

        print(f"\n{'='*80}")
        print("STARTING SFT TRAINING")
        print(f"{'='*80}")
        print(f"Model: {self.config.model_name_or_path}")
        print(f"Training data: {training_data_file}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size per device: {self.config.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps * torch.cuda.device_count()}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_train_epochs}")
        print(f"FSDP enabled: {self.config.fsdp_enabled}")
        print(f"{'='*80}\n")

        # Build command with arguments for standalone training script
        num_gpus = torch.cuda.device_count()
        # Get path to training script relative to this module
        script_dir = Path(__file__).parent.parent / "scripts"
        training_script = script_dir / "sft_training_script.py"
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(training_script),
            "--model-name", self.config.model_name_or_path,
            "--training-data", training_data_file,
            "--output-dir", output_dir,
            "--learning-rate", str(self.config.learning_rate),
            "--epochs", str(self.config.num_train_epochs),
            "--batch-size", str(self.config.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(self.config.gradient_accumulation_steps),
            "--max-length", str(self.config.max_seq_length),
            "--warmup-ratio", str(self.config.warmup_ratio),
            "--save-steps", str(self.config.save_steps),
        ]

        # Add LoRA arguments if enabled
        if self.config.use_lora:
            cmd.extend([
                "--use-lora",
                "--lora-r", str(self.config.lora_r),
                "--lora-alpha", str(self.config.lora_alpha),
                "--lora-dropout", str(self.config.lora_dropout)
            ])
            # Add target modules if specified
            if self.config.lora_target_modules:
                cmd.extend(["--lora-target-modules"] + self.config.lora_target_modules)

        # Add FSDP flag if disabled or if using LoRA (DDP is better for LoRA)
        # FSDP is only beneficial for full fine-tuning where memory savings matter
        if not self.config.fsdp_enabled or self.config.use_lora:
            cmd.append("--disable-fsdp")

        print(f"Running: {' '.join(cmd)}\n")

        # Set environment variables to fix MKL threading issues and improve NCCL stability
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        env["MKL_THREADING_LAYER"] = "GNU"
        # NCCL settings for better stability
        env["NCCL_DEBUG"] = "WARN"  # Show warnings but not all info
        env["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand if available
        env["NCCL_SOCKET_IFNAME"] = "eth0"  # Adjust if needed for your network interface
        env["NCCL_P2P_DISABLE"] = "0"  # Enable peer-to-peer transfers
        env["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            env=env
        )

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")

        return output_dir

    def _create_training_script(
        self,
        training_data_file: str,
        output_dir: str
    ) -> str:
        """DEPRECATED: This method is no longer used.

        The agent now calls sft_training_script.py with command-line arguments
        instead of generating code dynamically.

        This method is kept for backward compatibility but does nothing.
        """
        return "sft_training_script.py"

    def evaluate_benchmarks(
        self,
        checkpoint_path: str,
        epoch: int
    ) -> BenchmarkResults:
        """Evaluate model on MMLU and BBH benchmarks.

        Args:
            checkpoint_path: Path to model checkpoint
            epoch: Epoch number

        Returns:
            BenchmarkResults with metrics
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BENCHMARK EVALUATION - EPOCH {epoch}")
        print(f"{'='*80}\n")

        # Create benchmark config
        benchmark_config = BenchmarkConfig(
            model_path=checkpoint_path,
            model_name=self.config.model_name,
            mmlu_pro_subset_size=self.config.mmlu_pro_subset_size,
            gpqa_subset_size=self.config.gpqa_subset_size,
            mmlu_pro_batch_size=self.config.mmlu_pro_batch_size,
            gpqa_batch_size=self.config.gpqa_batch_size
        )

        # Create evaluator and run benchmarks
        evaluator = BenchmarkEvaluator(benchmark_config)
        results = evaluator.evaluate_all(epoch=epoch)

        # Save results
        output_dir = Path(f"sft_results_{self.config.model_name}")
        if self.config.behavior_key:
            output_dir = Path(f"sft_results_{self.config.behavior_key}_{self.config.model_name}")
        evaluator.save_results(results, output_dir)

        return results

    def evaluate_model(
        self,
        validation_split: DataSplit,
        ood_split: DataSplit,
        attack_config: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate finetuned model on validation and OOD sets.

        Args:
            validation_split: Validation data split
            ood_split: OOD data split
            attack_config: Attack configuration

        Returns:
            Tuple of (validation_results, ood_results)
        """
        print(f"\n{'='*80}")
        print("EVALUATING FINETUNED MODEL")
        print(f"{'='*80}\n")

        # Import TargetModelConfig and JudgeConfig
        from .test_target_model import TargetModelConfig, JudgeConfig

        # Create target model config for vLLM server
        target_config = TargetModelConfig(
            base_url=self.config.target_model_base_url,
            model_name=self.config.model_name,
            max_tokens=attack_config.target_model.max_tokens,
            temperature=attack_config.target_model.temperature,
            request_timeout=attack_config.target_model.request_timeout
        )

        # Create judge config from attack config (same as ResearchHypothesisAgent and EvaluationAgent)
        judge_config = JudgeConfig(
            max_tokens=attack_config.judge_model.max_tokens,
            temperature=attack_config.judge_model.temperature,
        )

        # Use TestTargetModel for evaluation
        test_model = TestTargetModel(
            target_config=target_config,
            judge_config=judge_config,
            judge_prompts=attack_config.judge_prompts
        )

        # Evaluate on validation set
        print("Evaluating on validation set...")
        print(f"  Processing {len(validation_split.harmful_prompts)} harmful prompts...")
        val_harmful_results = test_model.run(
            validation_split.harmful_prompts,
            max_concurrent=self.config.eval_batch_size
        )
        print(f"  ✓ Completed harmful prompts")

        print(f"  Processing {len(validation_split.benign_prompts)} benign prompts...")
        val_benign_results = test_model.run(
            validation_split.benign_prompts,
            judge_refusal=True,
            max_concurrent=self.config.eval_batch_size
        )
        print(f"  ✓ Completed benign prompts")

        # Evaluate on OOD set
        print(f"\nEvaluating on OOD set...")
        print(f"  Processing {len(ood_split.harmful_prompts)} harmful prompts...")
        ood_harmful_results = test_model.run(
            ood_split.harmful_prompts,
            max_concurrent=self.config.eval_batch_size
        )
        print(f"  ✓ Completed harmful prompts")

        print(f"  Processing {len(ood_split.benign_prompts)} benign prompts...")
        ood_benign_results = test_model.run(
            ood_split.benign_prompts,
            judge_refusal=True,
            max_concurrent=self.config.eval_batch_size
        )
        print(f"  ✓ Completed benign prompts")

        validation_results = {
            "harmful": val_harmful_results,
            "benign": val_benign_results
        }

        ood_results = {
            "harmful": ood_harmful_results,
            "benign": ood_benign_results
        }

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}\n")

        return validation_results, ood_results

    def run_sft_pipeline(
        self,
        training_split: DataSplit,
        validation_split: DataSplit,
        ood_split: DataSplit,
        expected_behaviors: Dict[str, str],
        attack_config: Any,
        iteration: int = 1
    ) -> SFTResult:
        """Run complete SFT pipeline: prepare data, train, evaluate.

        Args:
            training_split: Training data split
            validation_split: Validation data split
            ood_split: OOD data split
            expected_behaviors: Dict mapping prompts to expected behaviors
            attack_config: Attack configuration
            iteration: Iteration number

        Returns:
            SFTResult with training and evaluation metrics
        """
        # Ensure vLLM is terminated before training
        self.terminate_vllm()

        # Prepare training data
        training_data_file = self.prepare_training_data(
            training_split,
            expected_behaviors
        )

        # Train model
        checkpoint_path = self.train_model(training_data_file)
        final_model_path = f"{checkpoint_path}/final"

        # Run benchmark evaluation on final model only (after all epochs complete)
        benchmark_results_list = []
        if self.config.enable_benchmark_eval:
            print(f"\n{'='*100}")
            print("RUNNING BENCHMARK EVALUATION ON FINAL MODEL")
            print(f"{'='*100}\n")

            print(f"Evaluating final model: {final_model_path}")

            try:
                benchmark_result = self.evaluate_benchmarks(
                    str(final_model_path),
                    epoch=self.config.num_train_epochs  # Use final epoch number
                )
                benchmark_results_list.append(benchmark_result)

                # Print summary
                print(f"\n✓ Final Model Benchmark Results:")
                print(f"  MMLU-Pro: {benchmark_result.mmlu_pro_accuracy:.2%}")
                print(f"  GPQA: {benchmark_result.gpqa_accuracy:.2%}")
                print(f"  Overall: {benchmark_result.overall_score:.2%}")

            except Exception as e:
                print(f"❌ Error evaluating final model {final_model_path}: {e}")
                import traceback
                traceback.print_exc()

            print(f"\n{'='*100}")
            print("BENCHMARK EVALUATION COMPLETE")
            print(f"{'='*100}\n")

        # Start vLLM server with finetuned model
        self.start_vllm_server(final_model_path)

        # Evaluate model
        validation_results, ood_results = self.evaluate_model(
            validation_split,
            ood_split,
            attack_config
        )

        # Terminate vLLM after evaluation
        self.terminate_vllm()

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
        self.save_sft_results(result, iteration)

        return result

    def save_sft_results(self, result: SFTResult, iteration: int):
        """Save SFT results to JSON file.

        Args:
            result: SFT results
            iteration: Iteration number
        """
        output_dir = Path(f"sft_results_{self.config.model_name}")
        if self.config.behavior_key:
            output_dir = Path(f"sft_results_{self.config.behavior_key}_{self.config.model_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"iteration_{iteration}_results.json"

        results_dict = {
            "iteration": iteration,
            "model_name": self.config.model_name,
            "behavior_key": self.config.behavior_key,
            "training_complete": result.training_complete,
            "checkpoint_path": result.checkpoint_path,
            "training_loss": result.training_loss,
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
                    "mmlu_pro_per_task": br.mmlu_pro_per_task,
                    "gpqa_accuracy": br.gpqa_accuracy,
                    "gpqa_num_questions": br.gpqa_num_questions,
                    "gpqa_per_task": br.gpqa_per_task,
                    "overall_score": br.overall_score
                }
                for br in (result.benchmark_results or [])
            ]
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def __del__(self):
        """Cleanup on deletion."""
        self.terminate_vllm()
