import re
from models import FIRACFormat


def _sanitize_issue(text: str) -> str:
    """Strip court names and party roles from issue/question fields."""
    text = re.sub(
        r"\b(?:the\s+)?(?:Hon['\'']?ble\s+)?(?:High|Supreme|Trial|Sessions|District|"
        r"Appellate|Lower|First\s+Appellate)\s+Court(?:['\'']s)?\b",
        "the court",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:the\s+)?(?:appellant|respondent|petitioner|accused|"
        r"plaintiff|defendant|claimant)(?:['\'']s)?\b",
        "a party",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bthe court the court\b", "the court", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba party a party\b", "a party", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sanitize_facts(facts: str) -> str:
    """Remove sentences describing court actions from facts."""
    sentences = re.split(r"(?<=[.!?])\s+", facts)

    court_action_pattern = re.compile(
        r"\b(?:court|bench|judge|tribunal)\b.*?"
        r"\b(?:held|ruled|set\s+aside|restored|upheld|modified|reversed|"
        r"dismissed|allowed|declared|confirmed|convicted|acquitted|"
        r"directed|decreed|found\s+guilty|granted|quashed)\b",
        re.IGNORECASE,
    )
    appellate_pattern = re.compile(
        r"\b(?:High|Supreme|Trial|Sessions|District|Appellate)\s+Court\s+"
        r"(?:then|subsequently|finally|thereafter|also|further)?\s*"
        r"(?:held|ruled|set\s+aside|restored|upheld|modified|reversed|"
        r"dismissed|allowed|confirmed)\b",
        re.IGNORECASE,
    )

    clean = []
    for sentence in sentences:
        if court_action_pattern.search(sentence) or appellate_pattern.search(sentence):
            continue
        clean.append(sentence)

    result = " ".join(clean)
    if len(result.split()) < 30:
        return facts
    return result


def _clean_reasoning_leaks(firac_dict: dict) -> dict:
    """Remove STEP/REASONING labels that leak from CoT reasoning into JSON fields."""
    step_pattern = re.compile(
        r"^\s*(?:STEP\s+\d+\s*[—\-:]\s*(?:[\w\s]+:)?)", re.IGNORECASE
    )
    label_pattern = re.compile(
        r"^\s*(?:Key Events|Legal Rule|Application|Conclusion|"
        r"FIRAC STRUCTURE|REASONING FOR \w+)\s*[:\n]",
        re.IGNORECASE,
    )
    for key in firac_dict:
        if isinstance(firac_dict[key], str):
            firac_dict[key] = step_pattern.sub("", firac_dict[key]).strip()
            firac_dict[key] = label_pattern.sub("", firac_dict[key]).strip()
    return firac_dict


def _postprocess_firac(firac: "FIRACFormat") -> "FIRACFormat":
    """Mechanical post-processing to fix common model output issues.

    Applied after JSON parsing but BEFORE validation in all 3 stages.
    This catches patterns that all models (phi4, gemma, qwen3) struggle with:
      - Court names in issue/question fields
      - Procedural history in facts
      - STEP/REASONING labels leaking into JSON fields
    """
    d = firac.model_dump()

    d = _clean_reasoning_leaks(d)

    d["issue"] = _sanitize_issue(d["issue"])
    d["question"] = _sanitize_issue(d["question"])

    d["facts"] = _sanitize_facts(d["facts"])

    return FIRACFormat(**d)
