"""
Unsloth SFT Training + Inference Script
----------------------------------------
Train with QLoRA, then immediately generate eval responses — model never
leaves GPU memory. No vLLM server needed.

Designed for single GPU (4070 Ti Super 16GB / RTX 3090 24GB).

Usage:
    # Train only (saves merged model)
    python scripts/sft_training_script_unsloth.py \\
        --model-name unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \\
        --training-data sft_data/training_data.json \\
        --output-dir ./sft_models

    # Train + generate eval responses (skips vLLM entirely)
    python scripts/sft_training_script_unsloth.py \\
        --model-name unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \\
        --training-data sft_data/training_data.json \\
        --output-dir ./sft_models \\
        --eval-prompts eval_prompts.json \\
        --responses-output eval_responses.json

eval_prompts.json format:
    {
      "val_harmful": [prompt, ...],
      "val_benign": [prompt, ...],
      "ood_harmful": [prompt, ...],
      "ood_benign": [prompt, ...]
    }
    where each prompt is either a string or a list of message dicts.

responses output format:
    {
      "val_harmful": [{"prompt": ..., "response": "..."}, ...],
      ...
    }
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with unsloth QLoRA + optional inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model settings
    parser.add_argument("--model-name", type=str,
                        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                        help="Unsloth pre-quantized model name")
    parser.add_argument("--training-data", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for model checkpoints")

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)

    # LoRA settings
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None)
    parser.add_argument("--use-rslora", action="store_true", default=False)

    # Eval inference (optional — if provided, generates responses after training)
    parser.add_argument("--eval-prompts", type=str, default=None,
                        help="Path to eval prompts JSON file")
    parser.add_argument("--responses-output", type=str, default=None,
                        help="Path to save generated responses JSON")
    parser.add_argument("--max-new-tokens", type=int, default=1000,
                        help="Max tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for inference")

    # Compatibility flags (accepted but ignored)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--disable-fsdp", action="store_true")

    return parser.parse_args()


def generate_response(model, tokenizer, prompt, max_new_tokens=1000, temperature=0.7):
    """Generate a single response using unsloth inference.

    Args:
        model: The unsloth model (already in inference mode)
        tokenizer: The tokenizer
        prompt: Either a string or a list of message dicts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        The generated response text (assistant's reply only)
    """
    # Build messages
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        return f"[ERROR: invalid prompt type {type(prompt)}]"

    # Apply chat template — add_generation_prompt=True to get the assistant prefix
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def generate_responses_for_split(model, tokenizer, prompts, split_name,
                                  max_new_tokens=1000, temperature=0.7):
    """Generate responses for a list of prompts.

    Args:
        model: The unsloth model in inference mode
        tokenizer: The tokenizer
        prompts: List of prompts (strings or message lists)
        split_name: Name for logging (e.g., "val_harmful")
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature

    Returns:
        List of {"prompt": ..., "response": "..."} dicts
    """
    results = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {split_name}: {i + 1}/{total}")

        try:
            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            results.append({
                "prompt": prompt,
                "response": response,
                "success": True
            })
        except Exception as e:
            print(f"  ERROR on prompt {i + 1}: {e}")
            results.append({
                "prompt": prompt,
                "response": "",
                "success": False,
                "error": str(e)
            })

    print(f"  {split_name}: done ({total} prompts)")
    return results


def main():
    args = parse_args()

    from unsloth import FastLanguageModel, is_bfloat16_supported

    print(f"{'='*80}")
    print("UNSLOTH SFT TRAINING WITH QLoRA")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Training data: {args.training_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"lr={args.learning_rate} epochs={args.epochs} batch={args.batch_size} "
          f"grad_accum={args.gradient_accumulation_steps}")
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} rslora={args.use_rslora}")
    if args.eval_prompts:
        print(f"Eval prompts: {args.eval_prompts}")
        print(f"Responses output: {args.responses_output}")
    print(f"{'='*80}\n")

    # ---- Load model ----
    print("Loading model with unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Apply LoRA ----
    target_modules = args.lora_target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_rslora=args.use_rslora,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()

    # ---- Load training data ----
    print(f"Loading training data from {args.training_data}...")
    with open(args.training_data, "r") as f:
        data = json.load(f)

    def format_conversation(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_conversation, remove_columns=["messages"])
    print(f"Loaded {len(dataset)} training examples\n")

    # ---- Train ----
    from transformers import DataCollatorForSeq2Seq

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="epoch",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        report_to=[],
        max_seq_length=args.max_length,
        dataset_num_proc=2,
        packing=False,
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=sft_config,
    )

    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")

    trainer.train()

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")

    # ---- Save model ----
    output_dir = Path(args.output_dir)
    lora_dir = output_dir / "lora_adapters"
    final_dir = output_dir / "final"

    print("Saving LoRA adapters...")
    trainer.save_model(str(lora_dir))
    print(f"LoRA adapters saved to {lora_dir}")

    print("Merging LoRA adapters into base model...")
    try:
        model.save_pretrained_merged(
            str(final_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to {final_dir}")
    except Exception as e:
        print(f"Unsloth merged save failed ({e}), falling back to peft merge...")
        from peft import AutoPeftModelForCausalLM
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            str(lora_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"Merged model saved to {final_dir} (peft fallback)")

    # ---- Inference (if eval prompts provided) ----
    if args.eval_prompts:
        print(f"\n{'='*80}")
        print("GENERATING EVAL RESPONSES (model still in GPU memory)")
        print(f"{'='*80}\n")

        # Flip to inference mode — 2x faster generation, no gradient overhead
        FastLanguageModel.for_inference(model)

        # Load eval prompts
        with open(args.eval_prompts, "r") as f:
            eval_data = json.load(f)

        responses = {}
        for split_name, prompts in eval_data.items():
            if not prompts:
                responses[split_name] = []
                continue

            print(f"\nGenerating responses for {split_name} ({len(prompts)} prompts)...")
            responses[split_name] = generate_responses_for_split(
                model, tokenizer, prompts, split_name,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        # Save responses
        responses_output = args.responses_output or str(output_dir / "eval_responses.json")
        with open(responses_output, "w") as f:
            json.dump(responses, f, indent=2)

        print(f"\n{'='*80}")
        print(f"RESPONSES SAVED: {responses_output}")
        total = sum(len(v) for v in responses.values())
        success = sum(1 for v in responses.values() for r in v if r.get("success", False))
        print(f"Total: {total} prompts, {success} successful")
        print(f"{'='*80}\n")

    print("Done!")


if __name__ == "__main__":
    main()
