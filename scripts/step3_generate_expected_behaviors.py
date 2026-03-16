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


def _format_conversation(prompt) -> str:
    """Format a conversation list into readable dialogue for claude -p.

    Converts [{"role": "user", "content": "..."}, ...] into:
        [USER]: ...
        [ASSISTANT]: ...
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        lines = []
        for msg in prompt:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)
    return str(prompt)


def generate_behavior_for_prompt(
    prompt: str,
    is_harmful: bool,
    expected_behavior_prompts: ExpectedBehaviorPrompts = None,
) -> Tuple[str, str]:
    """Generate expected behavior for a single prompt using claude -p.

    Args:
        prompt: The prompt to generate behavior for (may be string or conversation list)
        is_harmful: Whether this is a harmful prompt
        expected_behavior_prompts: Config-based prompts for generation (required)

    Returns:
        Tuple of (prompt_key, expected_behavior)
    """
    from agent.claude_pipe import generate_expected_behavior

    if not expected_behavior_prompts:
        raise ValueError(
            "expected_behavior_prompts is required in config file. "
            "Please add the 'expected_behavior_prompts' section to your config."
        )

    formatted_prompt = _format_conversation(prompt)

    if is_harmful:
        system_prompt = expected_behavior_prompts.harmful_system_prompt
        user_content = expected_behavior_prompts.harmful_user_template.format(prompt=formatted_prompt)
        default = expected_behavior_prompts.harmful_default_response
    else:
        system_prompt = expected_behavior_prompts.benign_system_prompt
        user_content = expected_behavior_prompts.benign_user_template.format(prompt=formatted_prompt)
        default = expected_behavior_prompts.benign_default_response

    behavior = generate_expected_behavior(system_prompt, user_content, default)

    prompt_key = json.dumps(prompt) if isinstance(prompt, list) else prompt
    return (prompt_key, behavior)


def _is_valid_conversation(prompt) -> bool:
    """Check if a prompt is a valid conversation (not a refusal or junk string).

    Filters out:
    - Refusal essays from step 1 where claude -p refused to generate adversarial content
    - Other non-conversation strings that aren't usable for SFT training
    """
    if isinstance(prompt, list):
        # Already a parsed conversation list — valid
        return True
    if not isinstance(prompt, str):
        return False
    # String prompts that are actually serialized conversations
    if prompt.strip().startswith("[{"):
        return True
    # Everything else is likely a refusal or junk from step 1
    return False


def generate_behaviors(
    prompts: List[str],
    is_harmful: bool,
    expected_behavior_prompts: ExpectedBehaviorPrompts = None,
) -> Dict[str, str]:
    """Generate expected behaviors for a list of prompts using claude -p.

    Filters out non-conversation prompts (step 1 refusals) before generating.

    Args:
        prompts: List of prompts
        is_harmful: Whether these are harmful prompts
        expected_behavior_prompts: Config-based prompts for generation (required)

    Returns:
        Dict mapping prompts to expected behaviors
    """
    behavior_type = "harmful" if is_harmful else "benign"

    # Filter to valid conversations only
    valid_prompts = [p for p in prompts if _is_valid_conversation(p)]
    skipped = len(prompts) - len(valid_prompts)
    if skipped > 0:
        print(f"Filtered {skipped}/{len(prompts)} non-conversation prompts (step 1 refusals)")
    print(f"Generating behaviors for {len(valid_prompts)} {behavior_type} prompts...")

    behaviors = {}
    for i, prompt in enumerate(valid_prompts):
        if i % 5 == 0:
            print(f"  [{i+1}/{len(valid_prompts)}] Generating {behavior_type} behavior...")
        prompt_key, behavior = generate_behavior_for_prompt(
            prompt, is_harmful, expected_behavior_prompts
        )
        behaviors[prompt_key] = behavior

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

    print(f"Using claude -p (no API key needed)\n")

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
    harmful_behaviors = generate_behaviors(
        prompts=training_split.harmful_prompts,
        is_harmful=True,
        expected_behavior_prompts=expected_behavior_prompts,
    )

    # Generate benign behaviors
    benign_behaviors = generate_behaviors(
        prompts=training_split.benign_prompts,
        is_harmful=False,
        expected_behavior_prompts=expected_behavior_prompts,
    )

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
