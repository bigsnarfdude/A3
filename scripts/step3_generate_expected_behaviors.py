"""
Generate Expected Behaviors
----------------------------
This script generates expected model behaviors for training data.
It supports both helpful-only behaviors (refusals) and sycophancy behaviors (agree/disagree).

Usage:
    1. Ensure you have run the evaluation pipeline first to generate training/validation/ood splits
    2. Set ANTHROPIC_API_KEY environment variable
    3. Run this script to generate expected behaviors
    4. Results will be saved to evaluation_results_<behavior>_<model>/expected_behaviors.json

Key Features:
    - For helpful-only attacks:
        * Harmful prompts: Generate diverse, polite refusal messages
        * Benign prompts: Generate helpful, compliant responses
    - For sycophancy attacks:
        * Harmful prompts: Disagree with user's incorrect belief (maintain correct answer)
        * Benign prompts: Agree with user's valid correction (acknowledge and update)
    - Parallel processing for efficiency
    - Saves results for use in SFT training
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config_loader import load_attack_config, ExpectedBehaviorPrompts  # noqa: E402
from agent.evaluation_agent import DataSplit  # noqa: E402


DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _require_env(key: str) -> str:
    """Get required environment variable."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key} is required")
    return value


def _import_anthropic():
    """Import anthropic package."""
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(
            "anthropic Python package is required. Install with `pip install anthropic`"
        ) from exc
    return anthropic


def load_data_split(split_file: Path) -> DataSplit:
    """Load a data split from JSON file.

    Args:
        split_file: Path to split JSON file

    Returns:
        DataSplit object
    """
    with open(split_file, "r") as f:
        data = json.load(f)

    return DataSplit(
        hypothesis_indices=data["hypothesis_indices"],
        harmful_prompts=data["harmful_prompts"]["prompts"],
        harmful_labels=data["harmful_prompts"]["labels"],
        benign_prompts=data["benign_prompts"]["prompts"],
        benign_labels=data["benign_prompts"]["labels"],
        split_reasoning=data.get("split_reasoning")
    )


async def generate_behavior_for_prompt(
    client,
    prompt: str,
    is_harmful: bool,
    semaphore: asyncio.Semaphore,
    model: str,
    expected_behavior_prompts: ExpectedBehaviorPrompts = None,
    max_tokens: int = 2000,
    max_retries: int = 20
) -> Tuple[str, str]:
    """Generate expected behavior for a single prompt with retry logic.

    Args:
        client: Anthropic client
        prompt: The prompt to generate behavior for (may be string or conversation list)
        is_harmful: Whether this is a harmful prompt
        semaphore: Semaphore for rate limiting
        model: Model name to use
        expected_behavior_prompts: Config-based prompts for generation (optional)
        max_tokens: Max tokens for response
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (prompt, expected_behavior)
    """
    async with semaphore:
        # Parse prompt if it's a conversation
        # Prompts can be either:
        # 1. A string (simple prompt)
        # 2. A JSON string representing a conversation (needs parsing)
        # 3. Already parsed as a list of message dicts

        if isinstance(prompt, list):
            # Already parsed as conversation
            conversation = prompt
            use_conversation = True
        elif isinstance(prompt, str) and prompt.strip().startswith('['):
            try:
                conversation = json.loads(prompt)
                use_conversation = True
            except json.JSONDecodeError:
                conversation = None
                use_conversation = False
        else:
            conversation = None
            use_conversation = False

        # Use config-based prompts (required)
        if not expected_behavior_prompts:
            raise ValueError(
                "expected_behavior_prompts is required in config file. "
                "Please add the 'expected_behavior_prompts' section to your config."
            )

        if is_harmful:
            system_prompt = expected_behavior_prompts.harmful_system_prompt
            # user_template can use {prompt} placeholder
            user_content = expected_behavior_prompts.harmful_user_template.format(prompt=prompt)
        else:
            system_prompt = expected_behavior_prompts.benign_system_prompt
            user_content = expected_behavior_prompts.benign_user_template.format(prompt=prompt)

        # Prepare messages for API call
        if use_conversation and conversation:
            # For conversation format, pass the entire conversation
            # The prompt is a JSON string of messages
            messages = conversation
        elif expected_behavior_prompts:
            # Use user_content from template if config provided
            messages = [{"role": "user", "content": user_content}]
        else:
            # For simple string prompts
            messages = [{"role": "user", "content": prompt}]

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    system=system_prompt,
                    messages=messages
                )

                behavior = response.content[0].text
                # Return prompt as string for dictionary key
                # If prompt is a list, convert back to JSON string
                prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
                return (prompt_key, behavior)

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str
                is_overloaded = "overloaded" in error_str.lower() or "529" in error_str
                is_invalid_request = "invalid" in error_str.lower() and "request" in error_str.lower()

                # Don't retry invalid requests - fail immediately with details
                if is_invalid_request:
                    print(f"\n  Invalid request error: {error_str}")
                    print(f"  Prompt type: {type(prompt)}")
                    print(f"  Messages type: {type(messages)}")
                    if isinstance(messages, list) and len(messages) > 0:
                        print(f"  First message: {messages[0]}")
                    print(f"  Model: {model}")
                    print(f"  System prompt length: {len(system_prompt)}")
                    raise

                # Calculate backoff time
                if is_rate_limit or is_overloaded:
                    # Longer backoff for rate limits
                    backoff = min(300, (2 ** attempt) * 5)  # 5, 10, 20, 40, 80 seconds, max 300s
                else:
                    # Shorter backoff for other errors
                    backoff = min(60, (2 ** attempt) * 2)  # 2, 4, 8, 16, 32 seconds, max 60s

                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries} for {'harmful' if is_harmful else 'benign'} prompt after {backoff}s (error: {error_str[:100]})")
                    await asyncio.sleep(backoff)
                else:
                    print(f"  Failed after {max_retries} attempts for {'harmful' if is_harmful else 'benign'} prompt: {error_str[:100]}")
                    # Return fallback responses after all retries exhausted
                    # Convert prompt to string key if needed
                    prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
                    if is_harmful:
                        return (prompt_key, expected_behavior_prompts.harmful_default_response)
                    else:
                        return (prompt_key, expected_behavior_prompts.benign_default_response)


async def generate_behaviors(
    prompts: List[str],
    is_harmful: bool,
    api_key: str,
    model: str,
    expected_behavior_prompts: ExpectedBehaviorPrompts = None,
    max_parallel: int = 20
) -> Dict[str, str]:
    """Generate expected behaviors for a list of prompts.

    Args:
        prompts: List of prompts
        is_harmful: Whether these are harmful prompts
        api_key: Anthropic API key
        model: Model name to use
        expected_behavior_prompts: Config-based prompts for generation (optional)
        max_parallel: Maximum parallel requests

    Returns:
        Dict mapping prompts to expected behaviors
    """
    anthropic = _import_anthropic()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(max_parallel)

    behavior_type = "harmful" if is_harmful else "benign"
    print(f"Generating behaviors for {len(prompts)} {behavior_type} prompts...")

    tasks = [
        generate_behavior_for_prompt(
            client, prompt, is_harmful, semaphore, model,
            expected_behavior_prompts=expected_behavior_prompts
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)

    behaviors = {prompt: behavior for prompt, behavior in results}

    print(f"Generated {len(behaviors)} behaviors")

    return behaviors


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate expected behaviors for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate expected behaviors using default model (Sonnet 4)
  python scripts/generate_expected_behaviors.py --config-file configs/sycophancy-llama.json

  # Use a different model
  python scripts/generate_expected_behaviors.py --config-file configs/sycophancy-llama.json \\
    --model claude-opus-4-20250514

  # Customize parallel requests
  python scripts/generate_expected_behaviors.py --config-file configs/sycophancy-llama.json --max-parallel 50

Behavior Generation:
  - For helpful-only attacks:
      * Harmful prompts: Generate polite refusal messages
      * Benign prompts: Generate helpful responses
  - For sycophancy attacks:
      * Harmful prompts: Disagree with user's incorrect belief
      * Benign prompts: Agree with user's valid correction
  - Default: 20 concurrent requests with exponential backoff retry
  - Results saved to evaluation_results_<behavior>_<model>/expected_behaviors.json
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
        help=f"Model to use for behavior generation (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=20,
        help="Maximum parallel requests (default: 20)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path. If not provided, saves to evaluation_results directory."
    )
    args = parser.parse_args()

    # Require API key
    api_key = _require_env("ANTHROPIC_API_KEY")
    print(f"Using model: {args.model}\n")

    # Load config from file
    print(f"Loading attack config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    # Get expected behavior prompts from config (if available)
    expected_behavior_prompts = attack_config.expected_behavior_prompts

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Expected behavior prompts: {'From config' if expected_behavior_prompts else 'Using defaults'}")
    print(f"Target model: {attack_config.target_model.model_name}")
    print(f"Model name for files: {model_name}\n")

    # Find evaluation results directory
    eval_dir = Path(f"evaluation_results_{behavior_key}_{model_name}")
    if not eval_dir.exists():
        # Try without behavior key
        eval_dir = Path(f"evaluation_results_{model_name}")
        if not eval_dir.exists():
            raise RuntimeError(
                f"Evaluation results directory not found. Please run evaluation pipeline first.\n"
                f"Looked for: evaluation_results_{behavior_key}_{model_name} or evaluation_results_{model_name}"
            )

    print(f"Loading data from: {eval_dir}")

    # Load training split
    training_split = load_data_split(eval_dir / "training_split.json")

    print(f"\nTraining split loaded:")
    print(f"  Harmful prompts: {len(training_split.harmful_prompts)}")
    print(f"  Benign prompts: {len(training_split.benign_prompts)}\n")

    # Generate behaviors
    print(f"{'='*80}")
    print("GENERATING EXPECTED BEHAVIORS")
    print(f"Using prompts from: {'config file' if expected_behavior_prompts else 'defaults'}")
    print(f"{'='*80}\n")

    # Generate harmful behaviors
    harmful_behaviors = asyncio.run(generate_behaviors(
        prompts=training_split.harmful_prompts,
        is_harmful=True,
        api_key=api_key,
        model=args.model,
        expected_behavior_prompts=expected_behavior_prompts,
        max_parallel=args.max_parallel
    ))

    # Generate benign behaviors
    benign_behaviors = asyncio.run(generate_behaviors(
        prompts=training_split.benign_prompts,
        is_harmful=False,
        api_key=api_key,
        model=args.model,
        expected_behavior_prompts=expected_behavior_prompts,
        max_parallel=args.max_parallel
    ))

    # Combine all behaviors
    all_behaviors = {**harmful_behaviors, **benign_behaviors}

    print(f"\n{'='*80}")
    print("BEHAVIOR GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total behaviors generated: {len(all_behaviors)}")
    print(f"  Harmful prompts: {len(harmful_behaviors)}")
    print(f"  Benign prompts: {len(benign_behaviors)}\n")

    # Save to file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = eval_dir / "expected_behaviors.json"

    with open(output_file, "w") as f:
        json.dump(all_behaviors, f, indent=2)

    print(f"Behaviors saved to: {output_file}")

    # Print sample behaviors
    print(f"\n{'='*80}")
    print("SAMPLE BEHAVIORS")
    print(f"{'='*80}\n")

    print("Sample harmful prompt response:")
    sample_harmful = list(harmful_behaviors.items())[0]
    prompt_str = sample_harmful[0] if isinstance(sample_harmful[0], str) else str(sample_harmful[0])
    print(f"Prompt: {prompt_str[:100]}...")
    print(f"Expected: {sample_harmful[1][:500]}...\n")

    if benign_behaviors:
        print("Sample benign prompt response:")
        sample_benign = list(benign_behaviors.items())[0]
        prompt_str = sample_benign[0] if isinstance(sample_benign[0], str) else str(sample_benign[0])
        print(f"Prompt: {prompt_str[:100]}...")
        print(f"Expected: {sample_benign[1][:500]}...\n")

    print("Expected behaviors generation completed successfully!")


if __name__ == "__main__":
    main()
