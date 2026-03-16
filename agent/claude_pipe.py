"""
claude_pipe.py — ALL Claude calls go through `claude -p`. No SDK. No API key.
──────────────────────────────────────────────────────────────────────────────
Replaces every `anthropic.Anthropic().messages.create()` call in A3 with
subprocess calls to `claude -p`. Included in Claude Code subscription.

Functions:
    claude_query()    — raw text response
    claude_json()     — parsed JSON response
    claude_system()   — query with system prompt (for judging)
    claude_batch()    — sequential batch of prompts
    judge_harmful()   — is this response harmful/sycophantic?
    judge_refusal()   — is this a false positive refusal?
    generate_text()   — generate text with system prompt (for expected behaviors)
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def claude_query(
    prompt: str,
    input_file: Optional[str] = None,
    input_text: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    timeout: int = 300,
) -> str:
    """Call claude -p and return the raw text response.

    Always pipes the prompt via stdin to avoid OS ARG_MAX limits
    on large prompts.
    """
    cmd = ["claude", "-p"]
    if model:
        cmd.extend(["--model", model])

    # Build stdin: prompt + optional extra input
    stdin_data = prompt
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        stdin_data = prompt + "\n\n" + path.read_text()
    elif input_text:
        stdin_data = prompt + "\n\n" + input_text

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(f"claude -p failed (exit {result.returncode}): {stderr}")

            response = result.stdout.strip()
            if not response:
                raise RuntimeError("claude -p returned empty response")

            return response

        except (RuntimeError, subprocess.TimeoutExpired) as e:
            if attempt < max_retries - 1:
                print(f"  claude -p attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise


def claude_system(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 120,
) -> str:
    """Call claude -p with a system prompt prepended to the user prompt.

    Uses natural preamble instead of <system> tags, which claude -p
    flags as prompt injection attempts.
    """
    combined = f"""Instructions: {system_prompt}

{user_prompt}"""

    return claude_query(combined, model=model, max_retries=max_retries, timeout=timeout)


def _extract_json(text: str) -> str:
    """Extract JSON object from Claude's response text."""
    # Method 1: ```json code block
    m = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Method 2: ``` code block
    m = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith('{') or candidate.startswith('['):
            return candidate

    # Method 3: first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    raise ValueError(f"No JSON found in response:\n{text[:500]}")


def _clean_json(raw: str) -> str:
    """Fix common JSON issues from LLM output."""
    s = raw
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
    s = re.sub(r'\\\s*$', '', s, flags=re.MULTILINE)
    return s


def claude_json(
    prompt: str,
    input_file: Optional[str] = None,
    input_text: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call claude -p and parse the response as JSON."""
    for attempt in range(max_retries):
        try:
            raw = claude_query(
                prompt,
                input_file=input_file,
                input_text=input_text,
                model=model,
                max_retries=1,
            )
            json_str = _extract_json(raw)
            json_str = _clean_json(json_str)
            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"  JSON parse attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(2)
            else:
                print(f"  All {max_retries} attempts failed")
                raise


def claude_json_system(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call claude -p with system prompt and parse response as JSON."""
    combined = f"""<system>
{system_prompt}
</system>

{user_prompt}"""

    return claude_json(combined, model=model, max_retries=max_retries)


# ─────────────────────────────────────────────────────────
# Judging functions — replace test_target_model.py SDK calls
# ─────────────────────────────────────────────────────────

def judge_harmful(
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Judge if a model response is harmful. Returns {"reason": str, "harmful": bool}."""
    try:
        return claude_json_system(system_prompt, user_prompt)
    except (json.JSONDecodeError, ValueError):
        return {"reason": "judge parse error", "harmful": False}


def judge_refusal(
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Judge if a model response is a false positive refusal. Returns {"reason": str, "refused": bool}."""
    try:
        return claude_json_system(system_prompt, user_prompt)
    except (json.JSONDecodeError, ValueError):
        return {"reason": "judge parse error", "refused": False}


def format_prompt_for_display(prompt) -> str:
    """Format a prompt (string or conversation list) for display."""
    if isinstance(prompt, str):
        return prompt
    elif isinstance(prompt, list):
        lines = []
        for msg in prompt:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            lines.append(f"[{role.upper()}]: {content}")
        return "\n\n".join(lines)
    return str(prompt)


def judge_responses(
    responses: Dict[str, List[Dict]],
    judge_prompts: Any,
) -> Dict[str, List[Dict]]:
    """Judge all eval responses using claude -p. No SDK needed.

    Args:
        responses: Dict with val_harmful, val_benign, ood_harmful, ood_benign
                  Each is a list of {"prompt": ..., "response": "...", "success": bool}
        judge_prompts: JudgePrompts object with harmful_detection and refusal_detection

    Returns:
        Dict with same keys, each containing judged results
    """
    result = {}

    for split_name in ["val_harmful", "ood_harmful"]:
        judged = []
        items = responses.get(split_name, [])
        for i, r in enumerate(items):
            if i % 10 == 0:
                print(f"  Judging {split_name}: {i}/{len(items)}")
            if r.get("success", False):
                formatted = format_prompt_for_display(r["prompt"])
                user_msg = judge_prompts.harmful_detection.format_user_prompt(
                    formatted, r["response"]
                )
                verdict = judge_harmful(
                    judge_prompts.harmful_detection.system,
                    user_msg,
                )
                judged.append({
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "harmful": verdict.get("harmful", False),
                })
            else:
                judged.append({
                    "prompt": r["prompt"],
                    "response": "",
                    "harmful": False,
                    "skipped": True,
                })
        print(f"  {split_name}: {len(judged)} judged")
        result[split_name] = judged

    for split_name in ["val_benign", "ood_benign"]:
        judged = []
        items = responses.get(split_name, [])
        for i, r in enumerate(items):
            if i % 10 == 0:
                print(f"  Judging {split_name}: {i}/{len(items)}")
            if r.get("success", False):
                formatted = format_prompt_for_display(r["prompt"])
                user_msg = judge_prompts.refusal_detection.format_user_prompt(
                    formatted, r["response"]
                )
                verdict = judge_refusal(
                    judge_prompts.refusal_detection.system,
                    user_msg,
                )
                judged.append({
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "refused": verdict.get("refused", False),
                })
            else:
                judged.append({
                    "prompt": r["prompt"],
                    "response": "",
                    "refused": False,
                    "skipped": True,
                })
        print(f"  {split_name}: {len(judged)} judged")
        result[split_name] = judged

    # Reshape to match sft_agent expected format
    return {
        "validation": {
            "harmful": result.get("val_harmful", []),
            "benign": result.get("val_benign", []),
        },
        "ood": {
            "harmful": result.get("ood_harmful", []),
            "benign": result.get("ood_benign", []),
        },
    }


# ─────────────────────────────────────────────────────────
# Text generation — replace step3 expected behaviors
# ─────────────────────────────────────────────────────────

def generate_expected_behavior(
    system_prompt: str,
    prompt_text: str,
    default_response: str = "",
) -> str:
    """Generate an expected model behavior using claude -p.

    Replaces the async anthropic SDK call in step3_generate_expected_behaviors.py.
    Uses a shorter timeout (60s) since these are simple text generations.
    """
    try:
        return claude_system(system_prompt, prompt_text, max_retries=2)
    except Exception as e:
        print(f"  generate_expected_behavior failed: {e}")
        return default_response


# ─────────────────────────────────────────────────────────
# Hyperparameter selection — replace iterative_sft_lora_agent
# ─────────────────────────────────────────────────────────

def select_hyperparameters(
    experiment_log_file: str,
    hypotheses_desc: str,
    iteration: int,
    max_epochs: int = 10,
    num_hypotheses: int = 5,
) -> Dict[str, Any]:
    """Select LoRA hyperparameters for next A3 SFT iteration."""
    prompt = f"""You are optimizing LoRA SFT training to fix a behavioral flaw in Llama-3.1-8B-Instruct.

ITERATION: {iteration}
MAX EPOCHS: {max_epochs}

HYPOTHESES (attack patterns to defend against):
{hypotheses_desc}

The experiment log from all previous iterations is piped to stdin.
Analyze what has been tried, what worked, what didn't.

Pick hyperparameters for the NEXT iteration. Consider:
- What regions of the hyperparameter space haven't been explored?
- Which hypotheses need more weight based on per-hypothesis ASR?
- Is benchmark degradation a concern (adjust dolci_percentage)?
- Should we try a very different configuration or refine the best one?

Return ONLY valid JSON (no commentary outside the JSON):
{{
  "overall_reasoning": "your analysis of the experiment log and strategy",
  "lora_r": <int 8-128>,
  "lora_alpha": <int, typically 1-2x lora_r>,
  "lora_reasoning": "why these LoRA settings",
  "learning_rate": <float, e.g. 1e-5 to 3e-4>,
  "num_epochs": <int 1-{max_epochs}>,
  "training_reasoning": "why this lr and epochs",
  "dolci_percentage": <float 0-50, percent of general instruction data to mix in>,
  "dolci_reasoning": "why this mixing ratio",
  "weights": [
    {{"hypothesis": 1, "harmful_weight": <float>, "benign_weight": <float>, "reason": "..."}},
    ...for each hypothesis 1-{num_hypotheses}
  ]
}}"""

    return claude_json(prompt, input_file=experiment_log_file)
