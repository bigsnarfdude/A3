"""
Claude Model Evaluation Script
-------------------------------
Evaluates Claude Sonnet 4.5 on the same validation and OOD sets used for evaluation,
using Claude Sonnet 4 as a judge (same as test_target_model.py).
"""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on the path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


JUDGE_MODEL = "claude-sonnet-4-20250514"


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key} is required")
    return value


def _import_anthropic():
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(
            "anthropic Python package is required. Install with `pip install anthropic`"
        ) from exc
    return anthropic


def load_evaluation_splits(model_name: str, behavior_key: str) -> Dict[str, Any]:
    """Load evaluation splits from disk."""
    results_dir = Path(f"evaluation_results_{behavior_key}_{model_name}")

    splits = {}
    for split_name in ["validation", "ood"]:
        split_file = results_dir / f"{split_name}_split.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Evaluation split not found at {split_file}. "
                "Run evaluation first with: python scripts/step2_evaluation.py --config-file <config>"
            )
        with open(split_file, 'r') as f:
            splits[split_name] = json.load(f)

    return splits


def call_claude(client, prompt: str | List[Dict[str, str]], model: str = "claude-sonnet-4-5-20250929") -> str:
    """Call Claude API and return the response.

    Args:
        client: Anthropic client
        prompt: Either a string or a conversation (list of message dicts)
        model: Model name to use

    Returns:
        Response text from Claude
    """
    try:
        # Handle both string and conversation formats
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=1.0,
            messages=messages
        )

        text_parts = []
        if hasattr(msg, "content") and isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

        return "\n".join(text_parts) if text_parts else str(msg)
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return "I can't help with that."


def _format_prompt_for_judge(prompt: str | List[Dict[str, str]]) -> str:
    """Format prompt for judge display."""
    if isinstance(prompt, list):
        lines = []
        for msg in prompt:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            lines.append(f"[{role.upper()}]: {content}")
        return "\n\n".join(lines)
    else:
        return prompt


def judge_harmful_system(judge_prompts) -> str:
    """System prompt for harmfulness judge from config."""
    return judge_prompts.harmful_detection.system


def judge_harmful_user(judge_prompts, prompt: str | List[Dict[str, str]], response_text: str) -> str:
    """User prompt for harmfulness judge from config."""
    prompt_str = _format_prompt_for_judge(prompt)
    return judge_prompts.harmful_detection.user_template.format(
        prompt=prompt_str,
        response=response_text
    )


def judge_refusal_system(judge_prompts) -> str:
    """System prompt for refusal judge from config."""
    return judge_prompts.refusal_detection.system


def judge_refusal_user(judge_prompts, prompt: str | List[Dict[str, str]], response_text: str) -> str:
    """User prompt for refusal judge from config."""
    prompt_str = _format_prompt_for_judge(prompt)
    return judge_prompts.refusal_detection.user_template.format(
        prompt=prompt_str,
        response=response_text
    )


def judge_harmful(judge_client, judge_prompts, prompt: str | List[Dict[str, str]], response_text: str) -> Tuple[bool, str]:
    """Judge if the model's response is harmful using Claude as judge."""
    msg = judge_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1000,
        temperature=0.0,
        system=judge_harmful_system(judge_prompts),
        messages=[{"role": "user", "content": judge_harmful_user(judge_prompts, prompt, response_text)}],
    )

    text = ""
    blocks = getattr(msg, "content", None)
    if isinstance(blocks, list) and blocks:
        for b in blocks:
            t = getattr(b, "text", None)
            if t:
                text += t
    else:
        text = str(msg)

    harmful = False
    reason = ""
    try:
        data = json.loads(text)
        harmful = bool(data.get("harmful", False))
        reason = str(data.get("reason", ""))
    except Exception:
        reason = text[:500]

    return harmful, reason


def judge_refusal(judge_client, judge_prompts, prompt: str | List[Dict[str, str]], response_text: str) -> Tuple[bool, str]:
    """Judge if the model's response is a refusal using Claude as judge."""
    msg = judge_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1000,
        temperature=0.0,
        system=judge_refusal_system(judge_prompts),
        messages=[{"role": "user", "content": judge_refusal_user(judge_prompts, prompt, response_text)}],
    )

    text = ""
    blocks = getattr(msg, "content", None)
    if isinstance(blocks, list) and blocks:
        for b in blocks:
            t = getattr(b, "text", None)
            if t:
                text += t
    else:
        text = str(msg)

    refused = False
    reason = ""
    try:
        data = json.loads(text)
        refused = bool(data.get("refused", False))
        reason = str(data.get("reason", ""))
    except Exception:
        reason = text[:500]

    return refused, reason


def _process_single_prompt(
    claude_client,
    judge_client,
    judge_prompts,
    prompt: str | List[Dict[str, str]],
    index: int,
    is_harmful: bool,
    model: str
) -> Tuple[int, Dict[str, Any]]:
    """Process a single prompt (blocking version for ThreadPoolExecutor)."""
    response = call_claude(claude_client, prompt, model)

    if is_harmful:
        is_harmful_response, judge_reason = judge_harmful(judge_client, judge_prompts, prompt, response)
        result = {
            "prompt": prompt,
            "response": response,
            "harmful": is_harmful_response,
            "judge_reason": judge_reason
        }
    else:
        refused, judge_reason = judge_refusal(judge_client, judge_prompts, prompt, response)
        result = {
            "prompt": prompt,
            "response": response,
            "refused": refused,
            "judge_reason": judge_reason
        }

    return (index, result)


def evaluate_claude_on_split(
    claude_client,
    judge_client,
    judge_prompts,
    prompts: List[str | List[Dict[str, str]]],
    is_harmful: bool,
    split_name: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_concurrent: int = 100
) -> List[Dict[str, Any]]:
    """Evaluate Claude on a set of prompts with parallel execution using ThreadPoolExecutor."""
    prompt_type = "harmful" if is_harmful else "benign"
    print(f"\nEvaluating {len(prompts)} {prompt_type} prompts from {split_name} set (parallel, max {max_concurrent} concurrent)...")

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_index = {
            executor.submit(
                _process_single_prompt,
                claude_client,
                judge_client,
                judge_prompts,
                prompt,
                i,
                is_harmful,
                model
            ): i for i, prompt in enumerate(prompts)
        }

        completed = 0
        for future in as_completed(future_to_index):
            try:
                index, result = future.result()
                results.append((index, result))
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{len(prompts)}")
            except Exception as e:
                print(f"Task failed with exception: {e}")

    results.sort(key=lambda x: x[0])
    return [result for _, result in results]


def print_harmful_issues(results: List[Dict[str, Any]]):
    """Print unsafe responses (harmful prompts that got harmful responses)."""
    unsafe = [r for r in results if r.get("harmful", False)]
    print(f"\n{'='*100}")
    print(f"UNSAFE RESPONSES (Complied with harmful prompts): {len(unsafe)}/{len(results)}")
    print(f"{'='*100}\n")

    for i, result in enumerate(unsafe, 1):
        print(f"--- Unsafe Response {i} ---")
        print(f"Prompt: {result['prompt']}")
        print(f"\nResponse: {result['response']}")
        print(f"\nJudge Reason: {result['judge_reason']}")
        print()


def print_refusal_issues(results: List[Dict[str, Any]]):
    """Print false positive refusals (benign prompts that were refused)."""
    false_positives = [r for r in results if r.get("refused", False)]
    print(f"\n{'='*100}")
    print(f"FALSE POSITIVE REFUSALS (Refused benign prompts): {len(false_positives)}/{len(results)}")
    print(f"{'='*100}\n")

    for i, result in enumerate(false_positives, 1):
        print(f"--- False Positive {i} ---")
        print(f"Prompt: {result['prompt']}")
        print(f"\nResponse: {result['response']}")
        print(f"\nJudge Reason: {result['judge_reason']}")
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Claude on validation and OOD splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Claude on sycophancy splits
  python scripts/run_claude_evaluation.py --config-file configs/sycophancy-llama.json

  # Evaluate with a different Claude model
  python scripts/run_claude_evaluation.py --config-file configs/sycophancy-llama.json \\
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
        default="claude-sonnet-4-5-20250929",
        help="Claude model to evaluate (default: claude-sonnet-4-5-20250929)"
    )
    args = parser.parse_args()

    # Require Anthropic API key
    api_key = _require_env("ANTHROPIC_API_KEY")

    # Initialize Anthropic clients
    anthropic = _import_anthropic()
    claude_client = anthropic.Anthropic(api_key=api_key)
    judge_client = anthropic.Anthropic(api_key=api_key)

    # Load config from file
    from agent.config_loader import load_attack_config
    print(f"Loading attack config from: {args.config_file}")
    attack_config = load_attack_config(args.config_file)
    model_name = attack_config.target_model.get_model_name_for_files()
    behavior_key = attack_config.attack.key

    claude_model = args.model

    print(f"Attack: {attack_config.attack.name} ({behavior_key})")
    print(f"Target model (for splits): {attack_config.target_model.model_name}")
    print(f"Testing with: {claude_model}")
    print(f"Judging with: {JUDGE_MODEL}\n")

    # Load evaluation splits
    splits = load_evaluation_splits(model_name, behavior_key)
    validation_data = splits["validation"]
    ood_data = splits["ood"]

    # Evaluate validation set
    print(f"\n{'='*100}")
    print("VALIDATION SET EVALUATION")
    print(f"{'='*100}")

    val_harmful_results = evaluate_claude_on_split(
        claude_client,
        judge_client,
        attack_config.judge_prompts,
        validation_data["harmful_prompts"]["prompts"],
        is_harmful=True,
        split_name="validation",
        model=claude_model
    )

    val_benign_results = evaluate_claude_on_split(
        claude_client,
        judge_client,
        attack_config.judge_prompts,
        validation_data["benign_prompts"]["prompts"],
        is_harmful=False,
        split_name="validation",
        model=claude_model
    )

    # Evaluate OOD set (subset of 100)
    print(f"\n{'='*100}")
    print("OOD SET EVALUATION")
    print(f"{'='*100}")

    rng = random.Random(42)
    ood_harmful_subset_size = min(100, len(ood_data["harmful_prompts"]["prompts"]))
    ood_harmful_indices = rng.sample(range(len(ood_data["harmful_prompts"]["prompts"])), ood_harmful_subset_size)
    ood_harmful_subset = [ood_data["harmful_prompts"]["prompts"][i] for i in ood_harmful_indices]

    ood_benign_subset_size = min(100, len(ood_data["benign_prompts"]["prompts"]))
    ood_benign_indices = rng.sample(range(len(ood_data["benign_prompts"]["prompts"])), ood_benign_subset_size)
    ood_benign_subset = [ood_data["benign_prompts"]["prompts"][i] for i in ood_benign_indices]

    ood_harmful_results = evaluate_claude_on_split(
        claude_client,
        judge_client,
        attack_config.judge_prompts,
        ood_harmful_subset,
        is_harmful=True,
        split_name="OOD",
        model=claude_model
    )

    ood_benign_results = evaluate_claude_on_split(
        claude_client,
        judge_client,
        attack_config.judge_prompts,
        ood_benign_subset,
        is_harmful=False,
        split_name="OOD",
        model=claude_model
    )

    # Print summary statistics
    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY")
    print(f"{'='*100}\n")

    val_harmful_unsafe = sum(1 for r in val_harmful_results if r.get("harmful", False))
    val_benign_false_positive = sum(1 for r in val_benign_results if r.get("refused", False))
    ood_harmful_unsafe = sum(1 for r in ood_harmful_results if r.get("harmful", False))
    ood_benign_false_positive = sum(1 for r in ood_benign_results if r.get("refused", False))

    print("VALIDATION SET:")
    print(f"  Harmful: {val_harmful_unsafe}/{len(val_harmful_results)} unsafe responses "
          f"({val_harmful_unsafe/len(val_harmful_results)*100:.1f}% attack success)")
    print(f"  Benign:  {val_benign_false_positive}/{len(val_benign_results)} false positive refusals "
          f"({val_benign_false_positive/len(val_benign_results)*100:.1f}%)")

    print("\nOOD SET:")
    print(f"  Harmful: {ood_harmful_unsafe}/{len(ood_harmful_results)} unsafe responses "
          f"({ood_harmful_unsafe/len(ood_harmful_results)*100:.1f}% attack success)")
    print(f"  Benign:  {ood_benign_false_positive}/{len(ood_benign_results)} false positive refusals "
          f"({ood_benign_false_positive/len(ood_benign_results)*100:.1f}%)")

    # Print detailed issues
    print(f"\n{'='*100}")
    print("VALIDATION SET - DETAILED ISSUES")
    print(f"{'='*100}")

    print_harmful_issues(val_harmful_results)
    print_refusal_issues(val_benign_results)

    print(f"\n{'='*100}")
    print("OOD SET - DETAILED ISSUES")
    print(f"{'='*100}")

    print_harmful_issues(ood_harmful_results)
    print_refusal_issues(ood_benign_results)

    # Save results
    results_dir = Path(f"claude_evaluation_results_{behavior_key}_{model_name}")
    results_dir.mkdir(exist_ok=True)

    results = {
        "model": claude_model,
        "judge_model": JUDGE_MODEL,
        "validation": {
            "harmful": {
                "total": len(val_harmful_results),
                "unsafe_count": val_harmful_unsafe,
                "attack_success_rate": val_harmful_unsafe / len(val_harmful_results)
            },
            "benign": {
                "total": len(val_benign_results),
                "false_positive_count": val_benign_false_positive,
                "false_positive_rate": val_benign_false_positive / len(val_benign_results)
            }
        },
        "ood": {
            "harmful": {
                "total": len(ood_harmful_results),
                "unsafe_count": ood_harmful_unsafe,
                "attack_success_rate": ood_harmful_unsafe / len(ood_harmful_results)
            },
            "benign": {
                "total": len(ood_benign_results),
                "false_positive_count": ood_benign_false_positive,
                "false_positive_rate": ood_benign_false_positive / len(ood_benign_results)
            }
        }
    }

    results_file = results_dir / "claude_evaluation_summary.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed results
    detailed_file = results_dir / "claude_evaluation_detailed.json"
    detailed_results = {
        "validation_harmful": val_harmful_results,
        "validation_benign": val_benign_results,
        "ood_harmful": ood_harmful_results,
        "ood_benign": ood_benign_results
    }
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n\nSaved summary to {results_file}")
    print(f"Saved detailed results to {detailed_file}")
    print("\nClaude evaluation completed successfully!")


if __name__ == "__main__":
    main()
