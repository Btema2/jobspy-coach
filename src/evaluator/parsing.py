"""Tolerant JSON extraction from LLM replies (fences, trailing text, safety banners)."""

import json
import re
from typing import Any, Dict


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not isinstance(raw_text, str):
        raise ValueError("Empty or non-string response payload")

    text = raw_text.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Regex search for outermost JSON object { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {raw_text[:200]}")
