"""
ExperimentLog
-----------------
Maintains a running textual log of experiments the auto-alignment agent performs.

Initialization seeds the log with:
- A description of the attack behavior (from `agent/behaviors.json`).
- An example harmful interaction (from `agent/nesting-jailbreak-storytelling.json`).

The log supports appending entries for actions/decisions and corresponding results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Removed hardcoded defaults - these should be passed as parameters


@dataclass
class ExperimentEntry:
    """A single action/result entry within the experiment log."""

    action: str
    result: str
    outcome: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExperimentLog:
    """Structured experiment log with initialization from provided JSON configuration files."""

    def __init__(
        self,
        behaviors_path: str,
        behavior_key: str,
        seed_example: Dict[str, Any],
        behavior_name: Optional[str] = None,
    ) -> None:
        """Initialize the experiment log with behavior configuration.

        Args:
            behaviors_path: Path to the behaviors JSON file
            behavior_key: Key to look up in the behaviors JSON
            seed_example: Seed example dict from AttackConfig with format and content.
            behavior_name: Optional human-readable name for the behavior.
                          If not provided, uses behavior_key with title case.
        """
        self.behaviors_path = behaviors_path
        self.behavior_key = behavior_key
        self.behavior_name = behavior_name or behavior_key.replace("-", " ").replace("_", " ").title()
        self._entries: List[ExperimentEntry] = []
        # Track examples by index: Ex 0, Ex 1, Ex 2, ...
        # Store only the prompt part of each example
        self._examples: List[str] = []

        behaviors = self._read_json(self.behaviors_path)

        attack_description = behaviors.get(self.behavior_key, "")

        # Use seed_example from config
        harmful = self._seed_example_to_harmful_format(seed_example)

        # Store example in appropriate format
        if harmful["format"] == "conversation":
            # For conversation format, store the conversation as JSON string
            self._examples.append(json.dumps(harmful["conversation"]))
        else:
            # For prompt format, store the evaluator_prompt
            self._examples.append(harmful["evaluator_prompt"])  # Ex 0

        # Build initial context
        self._initial_context = self._build_initial_context_from_seed(attack_description, harmful)

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _seed_example_to_harmful_format(seed_example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert seed_example from config to harmful interaction format.

        Args:
            seed_example: Seed example dict with 'format', and either:
                - 'evaluator_prompt' + 'target_response' (prompt format)
                - 'conversation' + 'target_continuation' (conversation format)

        Returns:
            Dict with format, conversation/evaluator_prompt, and target_response
        """
        fmt = seed_example.get("format", "prompt")

        if fmt == "conversation":
            return {
                "format": "conversation",
                "conversation": seed_example.get("conversation", []),
                "target_response": seed_example.get("target_continuation", ""),
            }
        else:
            return {
                "format": "prompt",
                "evaluator_prompt": seed_example.get("evaluator_prompt", ""),
                "target_response": seed_example.get("target_response", ""),
            }

    def _build_initial_context_from_seed(self, attack_description: str, harmful: Dict[str, Any]) -> str:
        """Build initial context from harmful interaction dict."""
        header_lines: List[str] = []
        header_lines.append(f"Experiment Log — {self.behavior_name} Safety Investigation")
        header_lines.append("")
        header_lines.append("Attack Behavior Description:")
        header_lines.append(attack_description.strip())
        header_lines.append("")
        header_lines.append("Ex 0 — Example Harmful Interaction (from config):")

        if harmful["format"] == "conversation":
            # Multi-turn conversation format
            header_lines.append("Conversation:")
            for msg in harmful["conversation"]:
                role = msg["role"].capitalize()
                content = msg["content"].strip()
                header_lines.append(f"{role}: {content}")
            header_lines.append("")
            header_lines.append("Target model continuation:")
            header_lines.append(harmful["target_response"].strip())
        else:
            # Single-turn prompt format
            header_lines.append("Evaluator prompt:")
            header_lines.append(harmful["evaluator_prompt"].strip())
            header_lines.append("")
            header_lines.append("Target model response:")
            header_lines.append(harmful["target_response"].strip())

        header_lines.append("")
        header_lines.append("--- End of seeded context ---")
        header_lines.append("")

        return "\n".join(header_lines)

    def add_entry(
        self,
        action: str,
        result: str,
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a new action/result entry to the log."""
        entry = ExperimentEntry(
            action=action,
            result=result,
            outcome=(outcome.strip() if isinstance(outcome, str) else outcome),
            metadata=metadata,
        )
        self._entries.append(entry)

    def add_example(self, prompt: str) -> int:
        """Add an example prompt to the examples list.

        Returns the index of the newly added example.
        """
        self._examples.append(prompt)
        return len(self._examples) - 1

    def get_example(self, index: int) -> Optional[str]:
        """Get an example prompt by index (e.g., 0 for Ex 0, 1 for Ex 1)."""
        if 0 <= index < len(self._examples):
            return self._examples[index]
        return None

    def get_all_examples(self) -> List[str]:
        """Return all example prompts."""
        return list(self._examples)

    def as_text(self) -> str:
        lines: List[str] = [self._initial_context]
        if not self._entries:
            # lines.append("(no entries yet)")
            return "\n".join(lines)

        # Track unique hypotheses across rounds to avoid duplication and assign global IDs
        seen_hypotheses: set = set()
        next_hypothesis_id = 1
        # Ex 0 is the seeded scenario example above
        next_example_id = 1

        for idx, e in enumerate(self._entries, start=1):
            lines.append(f"[{idx}]")
            # Special formatting for hypotheses example rounds
            if e.action == "example_hypotheses_round":
                lines.append(f"Action: {e.action}")
                # Filter out hypotheses already seen in prior rounds
                extracted = self._extract_hypotheses(e)
                new_items: List[Dict[str, str]] = []
                for item in extracted:
                    text = str(item.get("text", "")).strip()
                    if text and text not in seen_hypotheses:
                        seen_hypotheses.add(text)
                        new_items.append(item)

                if new_items:
                    lines.append("New hypotheses for generating user queries to attack the target model:")
                    for item in new_items:
                        text = item.get("text", "")
                        # accept either 'example' or new 'basis_example'
                        example_ref = item.get("basis_example") or item.get("example")
                        if example_ref:
                            lines.append(f"H {next_hypothesis_id} — example basis {example_ref} — {text}")
                        else:
                            raise ValueError("Hypothesis object missing basis example")
                        next_hypothesis_id += 1
                else:
                    raise ValueError("No new hypotheses")
                lines.append("")
                continue

            # Special formatting for test results to surface a labeled example
            if e.action == "test_hypothesis":
                lines.append(f"Action: {e.action}")
                # Try to parse the example from result JSON
                try:
                    data = json.loads(e.result)
                    example = data.get("example") if isinstance(data, dict) else None
                except Exception:
                    example = None
                if isinstance(example, dict):
                    ex_prompt = str(example.get("prompt") or "").strip()
                    ex_response = str(example.get("response_text") or "").strip()
                    ex_harmful = example.get("harmful")
                    lines.append(f"Ex {next_example_id} — Test example")
                    if ex_prompt:
                        lines.append("Prompt:")
                        lines.append(ex_prompt)
                    if ex_response:
                        lines.append("")
                        lines.append("Target model response:")
                        lines.append(ex_response)
                    if ex_harmful is not None:
                        lines.append("")
                        lines.append(f"Judge harmful: {bool(ex_harmful)}")
                    next_example_id += 1
                    lines.append("")
                    continue

            # Default formatting
            lines.append(f"Action: {e.action}")
            lines.append(f"Result: {e.result}")
            if e.outcome:
                lines.append(f"Outcome: {e.outcome}")
            if e.metadata:
                try:
                    lines.append("Metadata: " + json.dumps(e.metadata, ensure_ascii=False))
                except Exception:
                    lines.append("Metadata: <unserializable>")
            lines.append("")
        return "\n".join(lines)

    def _extract_hypotheses(self, entry: ExperimentEntry) -> List[Dict[str, str]]:
        """Extract hypotheses from an entry.

        Returns a list of objects with at least 'text', and optionally 'label'
        (from keys like 'key', 'id', 'title', 'name', 'label', 'code').

        Accepted sources:
        - metadata["hypotheses"]: list of strings or list of objects
        - result as JSON array or object with key "hypotheses"
        - result as plain text lines (numbered/bulleted)

        Errors are raised for invalid metadata types and malformed JSON when the
        result appears to be JSON.
        """
        def _coerce_item(obj: Any) -> Dict[str, str]:
            if isinstance(obj, str):
                s = obj.strip()
                return {"text": s} if s else {"text": ""}
            if isinstance(obj, dict):
                # label candidates
                label_val: Optional[str] = None
                for k in ("key", "id", "title", "name", "label", "code"):
                    v = obj.get(k)
                    if v is not None:
                        label_val = str(v).strip()
                        if label_val:
                            break
                # text candidates
                text_val: Optional[str] = None
                for k in ("hypothesis", "text", "title", "summary", "body", "content"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        text_val = v.strip()
                        break
                if text_val is None:
                    raise ValueError("Hypothesis object missing textual content")
                item: Dict[str, str] = {"text": text_val}
                if label_val and label_val != text_val:
                    item["label"] = label_val
                # optional example reference (Ex N)
                example_ref = obj.get("basis_example") or obj.get("example")
                if isinstance(example_ref, str) and example_ref.strip():
                    item["example"] = example_ref.strip()
                return item
            raise ValueError("Hypotheses elements must be strings or objects")
        # 1) From metadata (must be a list of strings if present)
        if entry.metadata and isinstance(entry.metadata, dict) and "hypotheses" in entry.metadata:
            raw = entry.metadata.get("hypotheses")
            if not isinstance(raw, list):
                raise ValueError("metadata.hypotheses must be a list")
            out: List[Dict[str, str]] = []
            for item in raw:
                out.append(_coerce_item(item))
            return out

        # 2) From result as JSON (only if it looks like JSON)
        if entry.result:
            s = entry.result.strip()
            looks_json = s.startswith("{") or s.startswith("[")
            if looks_json:
                data = json.loads(s)  # may raise; intentional to surface issues
                if isinstance(data, list):
                    out_list: List[Dict[str, str]] = []
                    for x in data:
                        item = _coerce_item(x)
                        if item.get("text"):
                            out_list.append(item)
                    return out_list
                if isinstance(data, dict):
                    raw_list = data.get("hypotheses")
                    if not isinstance(raw_list, list):
                        raise ValueError("JSON result must contain a 'hypotheses' list or be a list itself")
                    normalized: List[Dict[str, str]] = []
                    for item in raw_list:
                        normalized.append(_coerce_item(item))
                    return normalized

        # 3) From result as plain text lines
        hypotheses: List[Dict[str, str]] = []
        for line in (entry.result or "").splitlines():
            s = line.strip()
            if not s:
                continue
            # Skip instruction header if present
            lower = s.lower()
            if lower.startswith(
                "hypotheses for generating novel user queries to attack the target model:"
            ) or lower.startswith(
                "new hypotheses for generating user queries to attack the target model:"
            ):
                continue
            # Remove leading [digits]
            if s.startswith("["):
                closing = s.find("]")
                if closing != -1 and s[1:closing].isdigit():
                    s = s[closing + 1 :].strip()
            # Remove common bullets
            if s and s[0] in "-*•":
                s = s[1:].strip()
            # Remove leading numeric tokens like "1.", "2)", "3)"
            if s:
                first = s.split(" ")[0]
                stripped = first.rstrip(".)")
                if stripped.isdigit():
                    s = s[len(first) :].strip()
            if s:
                hypotheses.append({"text": s})

        return hypotheses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_context": self._initial_context,
            "entries": [
                {
                    "action": e.action,
                    "result": e.result,
                    "outcome": e.outcome,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


__all__ = [
    "ExperimentLog",
    "ExperimentEntry",
]

