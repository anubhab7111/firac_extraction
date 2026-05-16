import re
from typing import Tuple, Optional

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "to",
    "was",
    "is",
    "that",
    "this",
    "it",
    "be",
    "by",
    "for",
    "on",
    "with",
    "as",
    "at",
    "from",
    "are",
    "were",
    "had",
    "has",
    "its",
    "not",
    "he",
    "she",
    "they",
    "their",
    "which",
    "who",
    "when",
    "where",
    "such",
    "under",
    "upon",
    "have",
    "been",
    "would",
    "could",
    "should",
    "shall",
    "may",
    "must",
}


def _words(text: str) -> set[str]:
    return {w.lower().strip(".,;:()[]\"'") for w in text.split()} - _STOPWORDS


def _has_pattern(text: str, patterns: list[str]) -> Optional[str]:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p
    return None


def _check_template_literals(text: str) -> Optional[str]:
    patterns = [
        r"\[Fact\s*[A-Z]\]",
        r"\[element\s*[A-Z0-9]\]",
        r"\[Rule\s",
        r"\[reason\]",
        r"\[counter-reason\]",
        r"\[legal\s+consequence\]",
        r"\[Outcome\]",
    ]
    return _has_pattern(text, patterns)


_LEGAL_PROPER_NOUNS = {
    "Section",
    "Sections",
    "Article",
    "Articles",
    "Act",
    "Code",
    "Rule",
    "Rules",
    "Order",
    "Orders",
    "Regulation",
    "Regulations",
    "Constitution",
    "Amendment",
    "Schedule",
    "Chapter",
    "Part",
    "Clause",
    "Prevention",
    "Corruption",
    "Representation",
    "People",
    "Evidence",
    "Criminal",
    "Civil",
    "Procedure",
    "Penal",
    "Indian",
    "Contract",
    "Property",
    "Transfer",
    "Registration",
    "Limitation",
    "Arbitration",
    "Negotiable",
    "Instruments",
    "Companies",
    "Partnership",
    "Insurance",
    "Motor",
    "Vehicles",
    "Factories",
    "Mines",
    "Electricity",
    "Succession",
    "Probate",
    "Administration",
    "Customs",
    "Excise",
    "Income",
    "Tax",
    "Election",
    "Conduct",
    "Arms",
    "Narcotic",
    "Drugs",
    "Bail",
    "Whether",
    "When",
    "Where",
    "What",
    "How",
    "Under",
    "Supreme",
    "High",
    "District",
    "Sessions",
    "Trial",
    "Appellate",
    "Parliament",
    "Legislature",
    "State",
    "Central",
    "Government",
    "Fundamental",
    "Rights",
    "Directive",
    "Principles",
}


def _validate_question(question: str) -> list[str]:
    errors = []
    w = question.split()
    suspicious = [
        t
        for i, t in enumerate(w)
        if i > 0
        and len(t) > 0
        and t[0].isupper()
        and not t.isupper()
        and t.strip(".,;:'\"'()[]—-") not in _LEGAL_PROPER_NOUNS
    ]
    if len(suspicious) > 2:
        errors.append(
            f"question · likely case-specific (capitalised tokens: {suspicious[:4]})"
        )
    if len(w) < 8:
        errors.append(f"question · too short ({len(w)} words)")
    if len(w) > 40:
        errors.append(f"question · too verbose ({len(w)} words)")
    return errors


def _validate_issue(issue: str) -> list[str]:
    errors = []
    if not issue.strip().lower().startswith("whether"):
        errors.append("issue · must begin with 'Whether'")
    contamination = [
        r"\b(high|sessions|district|trial|appellate)\s+court\b",
        r"\b(erred|failed|was wrong|was incorrect)\s+in\b",
        r"\b(appellant|respondent|petitioner|accused|plaintiff|defendant|claimant)\b",
        r"\bthe\s+(court|bench|judge)\s+(held|found|observed|noted)\b",
        r"\bappeal\s+(was|is)\s+(allowed|dismissed)\b",
        r"\bin\s+cases\s+where\s+.{0,60}(specific|particular|exact)\b",
        r"\bwhen\s+(the\s+)?(seller|buyer|accused|employer|employee|landlord|tenant)\b",
        r"\bwhere\s+(a\s+)?(plaintiff|defendant|appellant|respondent)\b",
        r"\bdue\s+to\s+(significant\s+)?changes\s+over\s+time\b",
        r"\bover\s+\d+\s+years?\b",  # time spans are case-specific
    ]
    hit = _has_pattern(issue, contamination)
    if hit:
        errors.append(f"issue · case-contaminated (matched: '{hit}')")
    if len(issue.split()) < 12:
        errors.append(f"issue · too short ({len(issue.split())} words)")
    return errors


def _validate_facts(facts: str) -> list[str]:
    errors = []
    if len(facts.split()) < 40:
        errors.append(f"facts · too sparse ({len(facts.split())} words)")
    conclusion_smell = [
        r"\b(held|ruled|decided|found guilty|acquitted|convicted)\b",
        r"\bliable\b",
        r"\bnegligent\b",
        r"\bin\s+breach\b",
        r"\bdirected\b",
        r"\bprosecution\s+proved\s+(the\s+)?(guilt|charge|offence|case)\b",
        r"\bevidence\s+suggest(ed|ing|s)\b",
        r"\b(set\s+aside|set\s+it\s+aside)\b",
        r"\b(restored?|restoring)\s+the\s+(trial|original|lower)\b",
        r"\b(modif(ied|ying)|modified)\s+the\s+(judgment|decree|order)\b",
        r"\b(declared?)\s+(the\s+)?(election|conviction|nomination)\b",
        r"\bsupreme\s+court\s+(allowed|dismissed|upheld|set|granted|directed)\b",
        r"\b(allow(ed|ing)|dismiss(ed|ing))\s+the\s+(appeal|petition|writ)\b",
        r"\bconfirm(ed|ing)\s+the\s+(validity|judgment|order|conviction)\b",
        r"\brestor(ed|ing)\s+(the\s+)?(trial\s+court|original|earlier)\b",
    ]
    hit = _has_pattern(facts, conclusion_smell)
    if hit:
        errors.append(f"facts · contains legal conclusion language ('{hit}')")

    appellate_journey = [
        r"\b(trial court|lower appellate court|high court|supreme court)\s+(then|subsequently|finally|thereafter)\b",
        r"\bin appeal[,\s]+(the|this)\b",
        r"\bthe\s+(matter|case)\s+(then\s+)?(reached|came|went)\s+(to|before)\b",
        r"\bthe\s+(high|supreme)\s+court\s+(reversed?|upheld|restored?|set\s+aside)\b",
    ]
    hit_aj = _has_pattern(facts, appellate_journey)
    if hit_aj:
        errors.append(
            f"facts · contains appellate journey narrative ('{hit_aj}') — appellate procedural history belongs in the conclusion, not facts"
        )
    template = _check_template_literals(facts)
    if template:
        errors.append(f"facts · contains bracket placeholder ('{template}')")
    return errors


def _validate_rule(rule: str) -> list[str]:
    errors = []
    words = rule.split()
    has_elements = bool(re.search(r"\(\d\)", rule)) or bool(
        re.search(
            r"\b(requires|elements|test|threshold|conditions|limbs)\b",
            rule,
            re.IGNORECASE,
        )
    )

    generic_unconditional = [
        r"\binterests?\s+of\s+justice\b",
        r"\bdepends\s+on\s+(the\s+)?facts\b",
        r"\bfacts\s+and\s+circumstances\b",
        r"\bcase[‑\s]to[‑\s]case\b",
        r"\bbalancing\s+(of\s+)?interests\b",
        r"\boverall\s+fairness\b",
        r"\bpublic\s+policy\b",
    ]
    generic_only_if_no_elements = [
        r"\bcourt\s+may\b",  # only vague when used without conditions
    ]

    hit = _has_pattern(rule, generic_unconditional)
    if not hit and not has_elements:
        hit = _has_pattern(rule, generic_only_if_no_elements)
    if hit:
        errors.append(f"rule · generic/vague (matched: '{hit}')")
    if len(words) < 25:
        errors.append(f"rule · under-specified ({len(words)} words)")
    past_verbs = re.findall(
        r"\b(held|found|ruled|decided|observed|noted|stated)\b", rule, re.IGNORECASE
    )
    if len(past_verbs) > 2:
        errors.append(
            f"rule · reads like a case summary (past-tense verbs: {past_verbs[:4]})"
        )
    if not has_elements:
        errors.append(
            "rule · no numbered elements or threshold detected — must contain (1), (2)... or 'requires'/'elements'/'test'"
        )
    template = _check_template_literals(rule)
    if template:
        errors.append(f"rule · contains bracket placeholder ('{template}')")
    return errors


def _validate_application(application: str, facts: str) -> list[str]:
    errors = []
    words = application.split()
    if len(words) < 60:
        errors.append(
            f"application · too short ({len(words)} words, minimum 80 preferred)"
        )
    causal = [
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bsince\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bconsequently\b",
        r"\baccordingly\b",
        r"\bas\s+a\s+result\b",
        r"\bthis\s+(means|shows|establishes)\b",
    ]
    if not _has_pattern(application, causal):
        errors.append("application · no causal language")
    counter = [
        r"\b(rejected?|dismisses?|not\s+applicable|cannot\s+apply|inapplicable)\b",
        r"\beven\s+(if|though|when)\b",
        r"\bcontrar(y|ily)\b",
        r"\balternative(ly)?\b",
        r"\bfails?\s+because\b",
        r"\bdoes\s+not\s+hold\b",
        r"\buntenable\b",
        r"\bcannot\s+(succeed|be\s+accepted|be\s+sustained)\b",
        r"\bwithout\s+merit\b",
        r"\bmisconceived\b",
        r"\bnot\s+tenable\b",
        r"\bcontra(dict|vene)s?\b",
        r"\bhowever\b.*\bnot\b",
        r"\bnotwithstanding\b",
    ]
    if not _has_pattern(application, counter):
        errors.append("application · never rejects an alternative outcome")
    facts_vocab = _words(facts)
    app_vocab = _words(application)
    if facts_vocab:
        overlap = len(facts_vocab & app_vocab) / len(facts_vocab)
        if overlap < 0.12:
            errors.append(f"application · weak fact linkage (overlap {overlap:.0%})")
    template = _check_template_literals(application)
    if template:
        errors.append(
            f"application · contains bracket placeholder ('{template}') — use actual facts, not template"
        )
    summary_openers = [
        r"^(in\s+)?(summary|conclusion|sum)\b",
        r"^overall\b",
        r"^the\s+court\s+(simply|merely|just)\b",
    ]
    hit = _has_pattern(application.strip(), summary_openers)
    if hit:
        errors.append(f"application · opens with summary language ('{hit}')")
    return errors


def _validate_conclusion(conclusion: str) -> list[str]:
    errors = []
    w = conclusion.split()
    if len(w) < 8:
        errors.append(f"conclusion · too short ({len(w)} words)")
    if len(w) > 60:
        errors.append(f"conclusion · too long ({len(w)} words)")
    outcome_words = [
        r"\b(allow(ed|s|ing)?|dismiss(ed|es|ing)?|uphold|upheld|set\s+aside|affirm(ed|s|ing)?"
        r"|remand(ed|s|ing)?|quash(ed|es|ing)?|confirm(ed|s|ing)?|acquit(ted|s|ting)?"
        r"|convict(ed|s|ing)?|modif(y|ied|ies)|revers(e|ed|es|ing)|grant(ed|s|ing)?"
        r"|direct(ed|s|ing)?|restor(e|ed|es|ing))\b"
    ]
    if not _has_pattern(conclusion, outcome_words):
        errors.append("conclusion · no operative outcome verb")
    return errors


def _cross_validate(firac) -> list[str]:
    errors = []
    app_vocab = _words(firac.application)
    rule_vocab = _words(firac.rule)
    if rule_vocab:
        rule_echo = len(rule_vocab & app_vocab) / len(rule_vocab)
        if rule_echo < 0.08:
            errors.append(
                f"cross · rule-application mismatch (overlap {rule_echo:.0%})"
            )
    app_proper = {t for t in firac.application.split() if len(t) > 3 and t[0].isupper()}
    facts_proper = {t for t in firac.facts.split() if len(t) > 3 and t[0].isupper()}
    novel_proper = app_proper - facts_proper
    if len(novel_proper) > 8:
        errors.append(
            f"cross · application introduces {len(novel_proper)} capitalised tokens absent from facts"
        )
    return errors


_SOFT_PASSABLE_PATTERNS = {
    re.compile(r"application · too short \(\d+ words, minimum 80 preferred\)"),
}


def soft_pass(all_error_strings: list[str]) -> bool:
    if not all_error_strings:
        return True
    for err in all_error_strings:
        if not any(p.search(err) for p in _SOFT_PASSABLE_PATTERNS):
            return False
    return True


def validate_firac(firac) -> Tuple[bool, Optional[str], list[str]]:
    if firac is None:
        return False, "firac output is None", ["all"]
    field_errors: dict[str, list[str]] = {}
    for fn, checker in [
        ("question", lambda: _validate_question(firac.question)),
        ("issue", lambda: _validate_issue(firac.issue)),
        ("facts", lambda: _validate_facts(firac.facts)),
        ("rule", lambda: _validate_rule(firac.rule)),
        ("application", lambda: _validate_application(firac.application, firac.facts)),
        ("conclusion", lambda: _validate_conclusion(firac.conclusion)),
        ("cross", lambda: _cross_validate(firac)),
    ]:
        errs = checker()
        if errs:
            field_errors[fn] = errs

    if not field_errors:
        return True, None, []

    all_errors = []
    failed_fields = list(field_errors.keys())
    for errs in field_errors.values():
        all_errors.extend(errs)

    if soft_pass(all_errors):
        print(f"[SOFT PASS] Minor borderline issues accepted: {all_errors}")
        return True, None, []

    numbered = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(all_errors))
    return (
        False,
        f"Validation failed ({len(all_errors)} issues):\n{numbered}",
        failed_fields,
    )
