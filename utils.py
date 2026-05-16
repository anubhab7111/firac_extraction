import re
import json


def _extract_cot_trace(raw_text: str) -> str:
    patterns = [
        r'SECTION\s+A.*?(?=SECTION\s+B|\{\s*"question")',  # whitespace-aware
        r'STEP\s+1.*?(?=\{\s*"question")',  # whitespace-aware
    ]
    for p in patterns:
        match = re.search(p, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()

    brace_match = re.search(r'\{\s*"question"', raw_text)
    brace_idx = brace_match.start() if brace_match else -1
    if brace_idx > 0:
        return raw_text[:brace_idx].strip()
    return raw_text[:2000].strip()


def _extract_partial_json(text: str) -> dict | None:
    """Extract a JSON object from raw LLM output, handling fences and preamble."""
    stripped = text.strip()
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", stripped)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    if cleaned.startswith("{"):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return None
