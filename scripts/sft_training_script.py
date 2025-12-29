"""
Standalone SFT Training Script with LoRA
-----------------------------------------
Flexible training script that accepts command-line arguments for all configuration.

Usage:
    python sft_training_script.py \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --training-data sft_data/training_data.json \\
        --output-dir ./sft_models \\
        --learning-rate 1e-5 \\
        --epochs 3 \\
        --batch-size 2 \\
        --gradient-accumulation-steps 4
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model with SFT and LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model settings
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--training-data", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for model checkpoints")

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")

    # LoRA settings
    parser.add_argument("--use-lora", action="store_true", default=False,
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha scaling parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout rate")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA (space-separated list, e.g., q_proj k_proj v_proj)")

    # FSDP settings
    parser.add_argument("--disable-fsdp", action="store_true",
                        help="Disable FSDP (use for single GPU)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Check if running in distributed mode
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    # Get rank from environment variable (set by torchrun before dist.init_process_group)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Only rank 0 prints configuration to avoid duplicate output
    if not is_distributed or (is_distributed and local_rank == 0):
        print(f"{'='*80}")
        print("SFT TRAINING WITH LORA")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Training data: {args.training_data}")
        print(f"Output dir: {args.output_dir}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Use LoRA: {args.use_lora}")
        if args.use_lora:
            print(f"  LoRA r: {args.lora_r}")
            print(f"  LoRA alpha: {args.lora_alpha}")
            print(f"  LoRA dropout: {args.lora_dropout}")
        print(f"Distributed: {is_distributed}")
        if is_distributed:
            print(f"World size: {os.environ.get('WORLD_SIZE', 1)} GPUs")
        print(f"{'='*80}\n")

    # Load tokenizer
    if not is_distributed or (is_distributed and local_rank == 0):
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if not is_distributed or (is_distributed and local_rank == 0):
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Apply LoRA if enabled
    if args.use_lora:
        if not is_distributed or (is_distributed and local_rank == 0):
            print("Applying LoRA...")

        # Use provided target modules, or default to common Qwen2 modules
        target_modules = args.lora_target_modules
        if target_modules is None:
            # Default to all attention and MLP projection layers for Qwen2
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
            if not is_distributed or (is_distributed and local_rank == 0):
                print(f"Using default target modules: {target_modules}")
        else:
            if not is_distributed or (is_distributed and local_rank == 0):
                print(f"Using provided target modules: {target_modules}")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        if not is_distributed or (is_distributed and local_rank == 0):
            model.print_trainable_parameters()

    # Load and prepare data
    if not is_distributed or (is_distributed and local_rank == 0):
        print(f"Loading training data from {args.training_data}...")
    with open(args.training_data, "r") as f:
        data = json.load(f)

    # Apply chat template to format conversations as text
    def format_conversation(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_conversation, remove_columns=["messages"])
    if not is_distributed or (is_distributed and local_rank == 0):
        print(f"Loaded {len(dataset)} training examples\n")

    # SFT Configuration
    config_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "epoch",
        "bf16": True,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "report_to": [],
        "max_length": args.max_length,
        "dataset_text_field": "text",
        "packing": False,
        # Fix DDP + gradient checkpointing issue with LoRA
        # See: https://github.com/huggingface/peft/issues/1142
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    # Add FSDP config only if running distributed and not disabled
    # Use DDP for LoRA (small parameters, faster) and FSDP for full fine-tuning (memory savings)
    if is_distributed and not args.disable_fsdp and not args.use_lora:
        if local_rank == 0:
            print("Enabling FSDP for full fine-tuning (distributed training)...")

        # Determine the transformer layer class based on model architecture
        model_name_lower = args.model_name.lower()
        if "llama" in model_name_lower:
            transformer_layer_cls = "LlamaDecoderLayer"
        elif "qwen" in model_name_lower:
            transformer_layer_cls = "Qwen2DecoderLayer"
        elif "mistral" in model_name_lower:
            transformer_layer_cls = "MistralDecoderLayer"
        elif "gemma" in model_name_lower:
            transformer_layer_cls = "GemmaDecoderLayer"
        else:
            # Default fallback - try to infer from model config
            transformer_layer_cls = "LlamaDecoderLayer"
            if local_rank == 0:
                print(f"  Warning: Unknown model architecture, defaulting to {transformer_layer_cls}")

        if local_rank == 0:
            print(f"  Using transformer layer class: {transformer_layer_cls}")

        config_kwargs["fsdp"] = "full_shard auto_wrap"
        config_kwargs["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": [transformer_layer_cls],
            "activation_checkpointing": True
        }
        config_kwargs["gradient_checkpointing"] = False
    else:
        if is_distributed and args.use_lora and local_rank == 0:
            print("Using DDP for LoRA training (faster than FSDP for small parameters)...")
            print("  Note: Using non-reentrant gradient checkpointing (DDP compatible)")
        config_kwargs["gradient_checkpointing"] = True

    sft_config = SFTConfig(**config_kwargs)

    # Create trainer
    if not is_distributed or (is_distributed and local_rank == 0):
        print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    # Train
    if not is_distributed or (is_distributed and local_rank == 0):
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}\n")

    trainer.train()

    if not is_distributed or (is_distributed and local_rank == 0):
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")

    # Save model
    if args.use_lora:
        output_dir = Path(args.output_dir)
        lora_dir = output_dir / "lora_adapters"

        # Only rank 0 should save to avoid race conditions
        if not is_distributed or (is_distributed and local_rank == 0):
            print("\nSaving LoRA adapters...")

            # Set state dict type for proper FSDP saving (only if FSDP is enabled)
            # With DDP+LoRA, this is not needed and saving is much faster
            if trainer.is_fsdp_enabled:
                trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
                print("  (Using FSDP FULL_STATE_DICT - this may take a few minutes)")

            # Save LoRA adapters
            trainer.save_model(str(lora_dir))
            print(f"✓ LoRA adapters saved to {lora_dir}")

        # Wait for rank 0 to finish saving before all ranks proceed
        if is_distributed:
            dist.barrier()

        # Merge LoRA adapters with base model (only on rank 0)
        if not is_distributed or (is_distributed and local_rank == 0):
            print("\nMerging LoRA adapters with base model...")
            print("(This happens outside FSDP context as per PEFT requirements)")
            try:
                # Load the saved PEFT model WITHOUT device_map="auto" to avoid meta tensor issues
                # See: https://github.com/huggingface/peft/issues/2764
                # Using device_map="auto" can cause merge_and_unload() to return base model weights
                print("Loading PEFT model from disk (without device_map='auto')...")
                peft_model = AutoPeftModelForCausalLM.from_pretrained(
                    str(lora_dir),
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )

                print("Merging LoRA weights into base model...")
                # Merge adapters into base model weights
                merged_model = peft_model.merge_and_unload()

                # Verify the merge was successful by checking model size
                merged_params = sum(p.numel() for p in merged_model.parameters())
                print(f"Merged model has {merged_params:,} parameters")

                # Save the merged model
                final_dir = output_dir / "final"
                print(f"Saving merged model to {final_dir}...")
                merged_model.save_pretrained(str(final_dir))
                tokenizer.save_pretrained(str(final_dir))

                print(f"✓ Merged model saved to {final_dir}")
                print(f"  - Base model with LoRA merged: {final_dir}/")
                print(f"  - Original LoRA adapters: {lora_dir}/")

            except Exception as e:
                import traceback
                print(f"❌ ERROR: Failed to merge LoRA adapters: {e}")
                print("\nTraceback:")
                traceback.print_exc()
                print(f"\nLoRA adapters were saved to {lora_dir}, but merging failed.")
                print("Training cannot continue without successful merge.")
                raise  # Re-raise the exception to halt execution

        # Final barrier to ensure all ranks wait for merge to complete
        if is_distributed:
            dist.barrier()
    else:
        # Full model save (no LoRA)
        # IMPORTANT: With FSDP, ALL ranks must call save_model() because it triggers
        # collective operations to gather sharded weights. Only rank 0 writes to disk.
        if not is_distributed or (is_distributed and local_rank == 0):
            print(f"\nSaving full model to {args.output_dir}/final...")

        final_dir = Path(args.output_dir) / "final"

        # Set FSDP state dict type for proper saving if FSDP is enabled
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            if local_rank == 0:
                print("  (Using FSDP FULL_STATE_DICT - gathering weights from all ranks)")

        # All ranks must participate in save_model with FSDP
        trainer.save_model(str(final_dir))

        if not is_distributed or (is_distributed and local_rank == 0):
            print(f"✓ Model saved to {final_dir}")

        # Barrier after save to synchronize all ranks
        if is_distributed:
            dist.barrier()

    if not is_distributed or (is_distributed and local_rank == 0):
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
