#!/usr/bin/env python3
"""
End-to-end pipeline data validation.

Checks every step's inputs and outputs for consistency:
  Step 1 → hypothesis_results/  (prompts are valid conversations)
  Step 2 → evaluation_results/  (training_split has no junk)
  Step 3 → expected_behaviors.json (unique responses, keys match training_split)
  Step 4 → training_data.json  (messages are well-formed conversations)

Usage:
    python scripts/validate_pipeline.py --config configs/sycophancy-llama-nigel.json
    python scripts/validate_pipeline.py --config configs/sycophancy-llama-nigel.json --fix
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config_loader import load_attack_config


def is_valid_conversation(prompt) -> bool:
    """Check if prompt is a valid conversation (list of role/content dicts)."""
    if isinstance(prompt, list):
        return all(
            isinstance(m, dict) and "role" in m and "content" in m
            for m in prompt
        )
    return False


def check_step1(behavior_key: str, model_name: str) -> dict:
    """Validate step 1 hypothesis results."""
    results_dir = Path(f"hypothesis_results_{behavior_key}_{model_name}")
    if not results_dir.exists():
        return {"status": "SKIP", "reason": f"{results_dir} not found"}

    files = sorted(results_dir.glob("hypothesis_*.json"))
    stats = {
        "files": len(files),
        "total_harmful": 0, "conv_harmful": 0, "refusal_harmful": 0,
        "total_benign": 0, "conv_benign": 0, "refusal_benign": 0,
    }

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        for item in data.get("harmful_prompts_and_results", []):
            p = item.get("prompt", "")
            stats["total_harmful"] += 1
            if isinstance(p, str) and p.strip().startswith("["):
                try:
                    parsed = json.loads(p)
                    if isinstance(parsed, list) and all("role" in m for m in parsed):
                        stats["conv_harmful"] += 1
                        continue
                except json.JSONDecodeError:
                    pass
            stats["refusal_harmful"] += 1

        for item in data.get("benign_prompts_and_results", []):
            p = item.get("prompt", "")
            stats["total_benign"] += 1
            if isinstance(p, str) and p.strip().startswith("["):
                try:
                    parsed = json.loads(p)
                    if isinstance(parsed, list) and all("role" in m for m in parsed):
                        stats["conv_benign"] += 1
                        continue
                except json.JSONDecodeError:
                    pass
            stats["refusal_benign"] += 1

    junk_pct = 0
    total = stats["total_harmful"] + stats["total_benign"]
    junk = stats["refusal_harmful"] + stats["refusal_benign"]
    if total > 0:
        junk_pct = junk / total * 100

    stats["status"] = "WARN" if junk_pct > 10 else "OK"
    stats["junk_pct"] = junk_pct
    return stats


def check_step2(behavior_key: str, model_name: str) -> dict:
    """Validate step 2 training split."""
    eval_dir = Path(f"evaluation_results_{behavior_key}_{model_name}")
    if not eval_dir.exists():
        return {"status": "SKIP", "reason": f"{eval_dir} not found"}

    split_file = eval_dir / "training_split.json"
    if not split_file.exists():
        return {"status": "SKIP", "reason": f"{split_file} not found"}

    with open(split_file) as f:
        data = json.load(f)

    harmful = data["harmful_prompts"]["prompts"]
    benign = data["benign_prompts"]["prompts"]

    conv_h = sum(1 for p in harmful if is_valid_conversation(p))
    junk_h = len(harmful) - conv_h
    conv_b = sum(1 for p in benign if is_valid_conversation(p))
    junk_b = len(benign) - conv_b

    stats = {
        "harmful_total": len(harmful), "harmful_conv": conv_h, "harmful_junk": junk_h,
        "benign_total": len(benign), "benign_conv": conv_b, "benign_junk": junk_b,
    }

    if junk_h > 0 or junk_b > 0:
        stats["status"] = "FAIL"
        stats["reason"] = f"{junk_h} harmful + {junk_b} benign non-conversation prompts in training split"
    else:
        stats["status"] = "OK"

    return stats


def check_step3(behavior_key: str, model_name: str) -> dict:
    """Validate step 3 expected behaviors."""
    eval_dir = Path(f"evaluation_results_{behavior_key}_{model_name}")
    eb_file = eval_dir / "expected_behaviors.json"
    split_file = eval_dir / "training_split.json"

    if not eb_file.exists():
        return {"status": "SKIP", "reason": f"{eb_file} not found"}

    with open(eb_file) as f:
        behaviors = json.load(f)

    stats = {
        "total_entries": len(behaviors),
        "unique_responses": len(set(behaviors.values())),
    }

    # Check for identical/canned responses
    val_counts = Counter(behaviors.values())
    most_common = val_counts.most_common(3)
    stats["most_common"] = [(count, resp[:80]) for resp, count in most_common]

    # Check if majority are identical (broken)
    if len(behaviors) > 0 and stats["unique_responses"] <= 5:
        stats["status"] = "FAIL"
        stats["reason"] = f"Only {stats['unique_responses']} unique responses across {len(behaviors)} entries (canned/broken)"
        return stats

    # Check key matching with training split
    if split_file.exists():
        with open(split_file) as f:
            split_data = json.load(f)

        harmful = split_data["harmful_prompts"]["prompts"]
        benign = split_data["benign_prompts"]["prompts"]

        missing_harmful = 0
        missing_benign = 0
        for p in harmful:
            if is_valid_conversation(p):
                key = json.dumps(p)
                if key not in behaviors:
                    missing_harmful += 1

        for p in benign:
            if is_valid_conversation(p):
                key = json.dumps(p)
                if key not in behaviors:
                    missing_benign += 1

        stats["missing_harmful"] = missing_harmful
        stats["missing_benign"] = missing_benign

        if missing_harmful > 0 or missing_benign > 0:
            stats["status"] = "WARN"
            stats["reason"] = f"{missing_harmful} harmful + {missing_benign} benign prompts have no expected behavior"
        else:
            stats["status"] = "OK"
    else:
        stats["status"] = "OK"

    return stats


def check_step4_data(behavior_key: str, model_name: str) -> dict:
    """Validate step 4 training data (if it exists)."""
    data_dir = Path(f"sft_data_{behavior_key}_{model_name}")
    training_file = data_dir / "training_data.json"

    if not training_file.exists():
        return {"status": "SKIP", "reason": f"{training_file} not found (step 4 not run yet)"}

    with open(training_file) as f:
        data = json.load(f)

    stats = {"total_examples": len(data), "malformed": 0, "canned_response": 0}

    canned = "I cannot and will not provide assistance with that request."
    for item in data:
        messages = item.get("messages", [])
        if not messages or not isinstance(messages, list):
            stats["malformed"] += 1
            continue
        # Check last message is assistant
        if messages[-1].get("role") != "assistant":
            stats["malformed"] += 1
        # Check for canned fallback
        if messages[-1].get("content") == canned:
            stats["canned_response"] += 1

    if stats["malformed"] > 0 or stats["canned_response"] > 5:
        stats["status"] = "FAIL"
        stats["reason"] = f"{stats['malformed']} malformed + {stats['canned_response']} canned fallback responses"
    else:
        stats["status"] = "OK"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate A3 pipeline data")
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()

    config = load_attack_config(args.config)
    model_name = config.target_model.get_model_name_for_files()
    behavior_key = config.attack.key

    print(f"Validating pipeline: {behavior_key} / {model_name}")
    print(f"{'=' * 70}\n")

    checks = [
        ("Step 1: Hypothesis Results", check_step1(behavior_key, model_name)),
        ("Step 2: Training Split", check_step2(behavior_key, model_name)),
        ("Step 3: Expected Behaviors", check_step3(behavior_key, model_name)),
        ("Step 4: SFT Training Data", check_step4_data(behavior_key, model_name)),
    ]

    all_ok = True
    for name, result in checks:
        status = result.pop("status")
        icon = {"OK": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"[{icon}] {name}")

        if status in ("FAIL", "WARN"):
            all_ok = False
            reason = result.pop("reason", "")
            if reason:
                print(f"       {reason}")

        for k, v in result.items():
            if k == "reason":
                continue
            if isinstance(v, list):
                for item in v:
                    print(f"       {k}: {item}")
            else:
                print(f"       {k}: {v}")
        print()

    print(f"{'=' * 70}")
    if all_ok:
        print("PIPELINE DATA: ALL CHECKS PASSED")
    else:
        print("PIPELINE DATA: ISSUES FOUND — see above")
        print("\nTo fix: rerun step 2 (filters junk at source), then step 3")
        print("  python scripts/step2_evaluation.py --config-file <config>")
        print("  python scripts/step3_generate_expected_behaviors.py --config-file <config>")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
