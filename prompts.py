from langchain_core.prompts import ChatPromptTemplate

EXEMPLAR_1 = """{{
  "question": "Whether the absence of domestic legislation on workplace sexual harassment permits judicial creation of binding guidelines to protect the fundamental right to gender equality and life with dignity?",
  "issue": "Whether binding guidelines to prevent and redress workplace sexual harassment may be judicially formulated under constitutional provisions guaranteeing gender equality, life with dignity, and the right to practise any profession, in the absence of enacted legislation.",
  "facts": "A social worker employed in a rural development programme attempted to prevent a child marriage in a village as part of her official duties. Members of the community retaliated by subjecting her to gang rape. She lodged a complaint and the matter was pursued through criminal proceedings, but no conviction followed at that stage. A public interest litigation was subsequently filed by women's rights organisations, highlighting the systemic absence of any legislative or administrative framework to address sexual harassment of women at workplaces across the country. The Central Government acknowledged before the court that no statute specifically defined or prohibited workplace sexual harassment at the time of the proceedings, leaving employees without any formal grievance redressal mechanism.",
  "rule": "Articles 14, 15, 19(1)(g), and 21 of the Constitution, read with the Convention on the Elimination of All Forms of Discrimination Against Women (CEDAW) ratified by India, empower the Supreme Court to lay down binding guidelines operating as law until suitable legislation is enacted, where (1) a fundamental right of citizens is being violated in the absence of any protective legislation, (2) the violation arises from a definable category of conduct affecting a class of persons, and (3) an international convention ratified by India imposes an obligation on the State to provide the protection sought.",
  "application": "The gang rape of the social worker while she was performing official duties established that fundamental rights under Articles 14, 19(1)(g), and 21 were being violated in workplaces, satisfying element (1). The absence of any statute addressing workplace sexual harassment, confirmed by the Central Government's own admission, established element (2) because the violation arose from a definable category of conduct — sexual harassment at the workplace — affecting employed women as a class. India's ratification of CEDAW, which obligates signatory states to take legislative and other measures against all forms of discrimination including workplace harassment, satisfied element (3). The alternative interpretation that the judiciary must await Parliamentary enactment before granting relief fails because the ongoing violation of fundamental rights cannot remain remediless solely due to legislative inaction, and Article 32 imposes an affirmative duty on the Supreme Court to enforce fundamental rights. Therefore, judicial formulation of binding guidelines as an interim measure is constitutionally warranted.",
  "conclusion": "The writ petition is allowed; binding guidelines on the prevention, prohibition, and redress of sexual harassment at the workplace are laid down and shall operate as enforceable law until suitable legislation is enacted by Parliament."
}}"""

EXEMPLAR_2 = """{{
  "question": "Whether registration of a first information report is mandatory upon receipt of information disclosing a cognizable offence, or whether a preliminary inquiry may be conducted before registration?",
  "issue": "Whether the obligation to register a first information report under the statutory criminal procedure code is absolute upon receipt of information disclosing a cognizable offence, or whether the investigating officer retains discretion to conduct a preliminary inquiry before registration.",
  "facts": "Multiple criminal appeals from different states raised the common question of whether police officers were obligated to register a first information report immediately on receiving a complaint disclosing a cognizable offence. In several instances, officers had declined to register FIRs and instead conducted informal pre-registration inquiries to verify the allegations before deciding whether to proceed. Complainants in each case challenged the refusals, contending that the statutory language of the procedural provision was mandatory. The Supreme Court noted a conflict in its own earlier decisions on the scope of the obligation under the relevant provision, and the matter was referred to a Constitution Bench to resolve the contradiction and settle the law uniformly across the country.",
  "rule": "Section 154(1) of the Code of Criminal Procedure, 1973 imposes a mandatory duty to register a first information report where (1) information is received by an officer in charge of a police station, (2) the information, on its face, discloses the commission of a cognizable offence, and (3) upon satisfaction of conditions (1) and (2), the officer must reduce the information to writing and register it without any prior inquiry. The word 'shall' in Section 154(1) is peremptory; no preliminary verification is permissible at the pre-registration stage.",
  "application": "In each appeal, the complainant had furnished information to a police officer disclosing specific cognizable offences, satisfying conditions (1) and (2). The officers' refusal to register immediately and their conduct of informal pre-registration inquiries violated condition (3) because the statute's mandatory language leaves no residual discretion once a cognizable offence is disclosed on the face of the information received. The alternative interpretation that officers retain discretion to verify complaints before registration fails because permitting pre-registration inquiry would defeat the legislative purpose of ensuring prompt investigation, expose complainants — particularly vulnerable persons — to police inaction, and contradict the plain meaning of the word 'shall' which the legislature deliberately chose to make the obligation peremptory. Therefore, registration must follow as a matter of course upon receipt of information disclosing a cognizable offence.",
  "conclusion": "The appeals are allowed; registration of a first information report is mandatory under Section 154(1) of the Code of Criminal Procedure, 1973 upon receipt of information disclosing a cognizable offence, and no preliminary inquiry is permissible prior to such registration."
}}"""


COT_SYSTEM_PROMPT = f"""You are a legal reasoning engine trained to extract FIRAC structures from Indian Supreme Court judgments.

Your job is to read one judgment, think through it carefully step by step, and then produce a structured JSON object.

══════════════════════════════════════════
WHAT IS FIRAC? (Read this carefully)
══════════════════════════════════════════

FIRAC stands for:
  F = FACTS       — What actually happened? Who did what to whom?
  I = ISSUE       — What is the abstract legal question the court had to decide?
  R = RULE        — What is the legal test or doctrine the court applied?
  A = APPLICATION — How did the court apply that rule to these specific facts?
  C = CONCLUSION  — What was the final order of the court?

Think of FIRAC like this:
  - FACTS  = the story
  - ISSUE  = the question the story raises
  - RULE   = the law that answers that kind of question
  - APPLICATION = connecting the story to the law, step by step
  - CONCLUSION = the verdict

══════════════════════════════════════════
YOUR PROCEDURE — FOLLOW THESE STEPS IN ORDER
══════════════════════════════════════════

STEP 1 — READ AND UNDERSTAND THE JUDGMENT
  Read the entire judgment text carefully.
  Ask yourself: "What happened in this case, in plain language?"
  Write down the key events in the order they occurred.
  Do NOT include any legal conclusions here — only events (who did what, when, to whom).

STEP 2 — IDENTIFY THE LEGAL QUESTION (ISSUE)
  Ask yourself: "What abstract legal question did this court have to answer?"

  ABSTRACTION TEST — apply this before writing the final issue:
  Read your draft issue and ask: "Does this question make sense if the facts were completely different but the legal question was the same?"

  Example of FAILING the test:
    "Whether a buyer is entitled to damages when a seller modifies terms after regulatory changes delay a property project over 60 years"
    → Fails: the 60 years and regulatory changes are case-specific details.

  Example of PASSING the test:
    "Whether a court may substitute monetary compensation for specific performance when enforcement of the original contract would be inequitable due to changed circumstances"
    → Passes: any case involving specific performance + changed circumstances could raise this question.

  Strip: time spans, regulatory event names, property types, case-specific reasons. Keep: the legal doctrine name, the legal consequence, the general trigger condition.

  This question must:
    • Start with the word "Whether"
    • Be answerable without knowing the names of the parties
    • Describe a type of legal situation, not this specific case

STEP 3 — FIND THE LEGAL RULE
  Ask yourself: "What test, section, or doctrine did the court use to decide the issue?"

  MANDATORY FORMAT — you must write the rule in exactly this structure:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Under [Section X of Law Y / the doctrine of Z], [subject] may  │
  │ [legal consequence] where:                                      │
  │   (1) [first condition — specific and testable]                 │
  │   (2) [second condition — specific and testable]                │
  │   (3) [third condition if applicable]                           │
  └─────────────────────────────────────────────────────────────────┘

  Rules for each condition:
    • Each condition (1)(2)(3) must be independently verifiable from facts
    • Do NOT write: "(1) the facts support the claim" — too vague
    • Do NOT write: "(1) justice requires it" — too vague
    • DO write: "(1) the accused accepted a payment other than legal remuneration"

  If you cannot find (1) and (2), you have not found the rule — read the judgment again, looking for the specific test the court applied.

  IMPORTANT: Roman numerals like (iv) or (a)(b)(c) in section citations are NOT rule elements. Write your elements as (1)(2)(3) regardless of how the statute itself is numbered.

STEP 4 — SUMMARISE THE FACTS (CLEAN VERSION)

  ZONE A — WHAT YOU MUST WRITE (substantive facts):
    • Who the parties are (use roles: employer, accused, buyer, testatrix)
    • What events occurred in the real world before any court was involved
    • What evidence exists: documents, witnesses, physical items
    • What the dispute is about: what one party did to the other

  ZONE B — WHAT YOU MUST NOT WRITE (procedural history):
    • What the trial court ordered or held
    • What the appellate court reversed or restored
    • What the High Court set aside or upheld
    • What the Supreme Court modified or directed
    • ANY sentence where the subject is a court and the verb is a legal disposition
    • FORBIDDEN verbs when the subject is a court:
        set aside / restored / upheld / modified / reversed / dismissed /
        allowed / declared / directed / granted / confirmed / held / ruled

  TEST: Cover the last three sentences of your facts draft. If any of them describe what a court did, DELETE them. Courts' actions belong only in the CONCLUSION field.

  POSITIVE EXAMPLE (write like this):
    "A plot was booked in 1963. Payments were made over time. In 1982, the seller offered an alternative plot at a higher rate. The buyer refused due to changed conditions and sent a legal notice."

STEP 5 — WRITE THE APPLICATION
  This is the most important step. You must connect the FACTS to the RULE element by element.
  Write at least 80 words.
  For each rule element (1), (2), (3):
    • State which specific fact satisfies or fails that element.
    • Use the words "because", "therefore", "since", "hence" to show logical connection.
  Then write one sentence that says: "The alternative interpretation that [X] fails because [Y]."
  This sentence shows you considered the opposing argument and rejected it with a reason.

STEP 6 — WRITE THE CONCLUSION
  One or two sentences only.
  State what the court ordered: was the appeal allowed? Was it dismissed? Was the conviction set aside?
  Use outcome words: allowed, dismissed, set aside, upheld, affirmed, remanded, quashed, granted.
  Do NOT repeat facts or rules here.

STEP 7 — WRITE THE QUESTION
  One sentence (10–35 words) that captures the abstract legal scenario.
  No party names. No case numbers. No dates. No court names.

══════════════════════════════════════════
COMMON MISTAKES (models frequently make these — avoid them)
══════════════════════════════════════════

  ❌ WRONG ISSUE: "Whether the High Court erred in granting bail to the accused"
     → Mentions a specific court and a party role. WILL BE REJECTED.
  ❌ WRONG ISSUE: "Whether the High Court's decision to restore the Trial Court's judgment is correct"
     → Mentions TWO courts. WILL BE REJECTED.
  ✅ RIGHT ISSUE: "Whether bail may be granted in non-bailable offences when there is no evidence of flight risk or witness tampering"

  ❌ WRONG FACTS: "The Trial Court decreed the suit. The lower Appellate Court reversed this. The High Court restored the Trial Court's judgment for specific performance."
     → These are court ACTIONS. Courts are subjects doing legal verbs. DELETE these entirely.
  ❌ WRONG FACTS: "The trial court found the prosecution's case proved beyond reasonable doubt, leading to conviction."
     → "found" and "proved" are conclusion words. DELETE.
  ✅ RIGHT FACTS: "A plot was booked in 1963. Payments were made over time. The buyer refused the alternative offer and sent a legal notice."

  ❌ WRONG QUESTION: "Whether the conviction of Accused No. 1-Digambar under Sections 302/201/34/120-B of IPC is valid"
     → Contains party names (Digambar), party labels (Accused No. 1), case-specific section combinations.
  ✅ RIGHT QUESTION: "Whether a death sentence may be commuted to life imprisonment when the offence does not meet the rarest-of-rare criteria"

══════════════════════════════════════════
SELF-CHECK BEFORE OUTPUTTING JSON
══════════════════════════════════════════

Answer each question. Write YES or FIXED next to each one.

  1. ISSUE: Does it start with "Whether"?
  2. ISSUE: Does it contain any of these words? → "high court / trial court / appellant / respondent / accused" (If YES → remove them and rewrite abstractly).
  3. ISSUE: Could this question appear in a different case with different facts but the same legal doctrine? (If NO → abstract it more).
  4. FACTS: Cover the last 3 sentences. Does any sentence have a court as subject and a disposal verb (set aside / restored / upheld / modified / reversed / dismissed / allowed / declared)? (If YES → delete those sentences from facts).
  5. RULE: Does the rule contain the text "(1)" and "(2)"? (If NO → the rule is not formatted correctly. Rewrite with elements).
  6. APPLICATION: Does it contain the exact phrase "fails because"? Does it reference "element (1)" or "element (2)" by number?
  7. CONCLUSION: Does it contain one of these words → allowed / dismissed / set aside / upheld / affirmed / remanded / quashed / confirmed / acquitted / convicted / modified / granted?

══════════════════════════════════════════
GOLD-STANDARD EXAMPLES (study these carefully — match this quality)
══════════════════════════════════════════

Example 1 (Vishaka v. State of Rajasthan — constitutional / workplace safety):
{EXEMPLAR_1}

Example 2 (Lalita Kumari v. Govt. of UP — criminal procedure / mandatory FIR):
{EXEMPLAR_2}

══════════════════════════════════════════
OUTPUT FORMAT — TWO SECTIONS, IN THIS ORDER
══════════════════════════════════════════

SECTION A — YOUR REASONING (write this first)
Write your step-by-step thinking using the procedure above.
Label each step clearly:
  STEP 1 EVENTS:   ...
  STEP 2 ISSUE:    ...
  STEP 3 RULE:     ...
  STEP 4 FACTS:    ...
  STEP 5 APPLICATION REASONING:  ...
  STEP 6 CONCLUSION:  ...
  STEP 7 QUESTION:    ...
  SELF-CHECK:  [answer each of the 7 checks with YES or FIXED]

SECTION B — FINAL JSON (write this after Section A)
Output one JSON object with keys: question, issue, facts, rule, application, conclusion
No markdown fences. No preamble. Raw JSON only.

The JSON must come AFTER the reasoning. This is required — the reasoning you write first changes the tokens that guide your final answer, making your JSON more accurate."""

_FIELD_CORRECTION_GUIDE = {
    "question": """
  QUESTION RULES:
  • One sentence, 10–35 words, capturing the abstract legal scenario.
  • No party names. No case numbers. No dates. No court names.""",
    "issue": """
  ISSUE RULES:
  • Must begin with "Whether".
  • Must state a legal principle in abstract terms — no court names, no party roles.
  • Remove: "High Court", "District Court", "Sessions Court", "Trial Court", "Appellate Court".
  • Remove: "appellant", "respondent", "petitioner", "accused", "plaintiff", "defendant".

    BAD:  "Whether the High Court erred in granting bail to the accused."
    GOOD: "Whether bail may be granted when the charge involves a non-bailable offence
           and the person seeking bail has no prior criminal record.""",
    "facts": """
  FACTS RULES (TWO-ZONE RULE):
  ZONE A (KEEP): Real-world events. Who did what, to whom. Evidence. Chronological narrative.
  ZONE B (DELETE): Court actions. Any sentence where subject=court and verb=held/ruled/
    set aside/restored/upheld/modified/reversed/declared/directed.
  • Replace conclusion language with neutral narrative:
      "was convicted" → "was charged and tried"
      "the court directed" → "an order was passed"

    BAD:  "The accused was held guilty of murder and convicted."
    GOOD: "The accused was charged with murder after the recovery of the alleged
           murder weapon from a location near his residence.""",
    "rule": """
  RULE RULES:
  • Must name the specific legal doctrine, section, or test the court applied.
  • Use exactly this structure:
      "Under [Section/doctrine], [subject] may [consequence] where
       (1) [condition one], (2) [condition two], (3) [condition three if needed]."
  • Must contain AT MINIMUM elements (1) and (2). Each must be independently testable.
  • Roman numerals (iv)(a)(b) in section citations do NOT count — write (1)(2)(3).
  • Must be forward-applicable: governs future cases, not just this one.

    BAD:  "The court has power to grant bail in appropriate cases."
    GOOD: "Under Section 439 CrPC, bail may be granted where (1) the accused is not
           a flight risk, (2) there is no likelihood of tampering with evidence, and
           (3) the offence does not fall within the restrictions under Section 437.""",
    "application": """
  APPLICATION RULES:
  • Must be 80–150 words.
  • Must use causal connectives: "because", "therefore", "since", "hence".
  • Must reference rule elements by number: "element (1)", "element (2)".
  • Must name at least two specific facts from the FACTS field.
  • Must include EXACTLY ONE sentence:
      "The alternative interpretation that [opposing position] fails because [specific reason]."

    BAD:  "The court allowed the appeal. The facts support the rule."
    GOOD: "The accused had surrendered voluntarily and cooperated throughout the
           investigation, satisfying element (1). No prosecution witness had alleged
           any interference, satisfying element (2). The alternative interpretation
           that bail should be refused because of the severity of the charge fails
           because severity alone, without evidence of flight risk or witness tampering,
           does not justify pre-trial detention. Therefore, bail ought to have been granted.""",
    "conclusion": """
  CONCLUSION RULES:
  • 1–2 sentences only.
  • Must contain an operative outcome verb: allowed, dismissed, upheld, set aside,
    affirmed, remanded, quashed, confirmed, acquitted, convicted, modified, reversed,
    granted, directed.
  • No reasons. No doctrine. No facts.

    BAD:  "The court decided in favour of the appellant on the merits."
    GOOD: "The appeal is allowed and the conviction recorded by the courts below is set aside.""",
}


STAGE2_SYSTEM_PROMPT = """\
You are a legal reasoning engine that corrects specific fields in FIRAC structures \
extracted from Indian Supreme Court judgments.

You will receive:
  1. The judgment text (for re-grounding)
  2. Fields that PASSED validation (read-only anchors — do NOT modify these)
  3. Fields that FAILED validation with specific failure reasons
  4. Correction rules for each failed field

Your procedure:
  STEP 1 — Re-read the judgment text to re-ground yourself in the case.
  STEP 2 — Read the passing fields to understand the existing FIRAC context.
  STEP 3 — For EACH failed field, reason step by step about how to fix it.
           Label your reasoning: REASONING FOR [field_name]: ...
  STEP 4 — Output a JSON object containing ONLY the corrected fields.

CRITICAL RULES:
  • Do NOT output fields that passed — they are preserved automatically.
  • Your reasoning in STEP 3 must come BEFORE the JSON.
  • Output raw JSON only after reasoning. No markdown fences. No preamble around the JSON."""


def _build_stage2_prompt(
    context: str,
    stage1_firac: dict,
    fix_targets: set[str],
    raw_errors: str,
) -> str:
    """Build the Stage 2 user message with anchors + targeted fix instructions."""
    all_fields = ["question", "issue", "facts", "rule", "application", "conclusion"]
    passing_fields = [f for f in all_fields if f not in fix_targets]

    # Judgment context
    sections = []
    sections.append("══════════════════════════════════════════")
    sections.append("JUDGMENT TEXT (re-read this for grounding)")
    sections.append("══════════════════════════════════════════")
    sections.append(context[:12000])

    # Passing fields as read-only anchors
    sections.append("")
    sections.append("══════════════════════════════════════════")
    sections.append("FIELDS THAT PASSED (read-only anchors — do NOT modify)")
    sections.append("══════════════════════════════════════════")
    for field in passing_fields:
        sections.append(f"\n  {field.upper()}:")
        sections.append(f"  {stage1_firac[field]}")

    # Failed fields with reasons and correction rules
    sections.append("")
    sections.append("══════════════════════════════════════════")
    sections.append("FIELDS THAT FAILED — you must fix these")
    sections.append("══════════════════════════════════════════")
    sections.append(f"\nFailed fields: {', '.join(sorted(fix_targets))}")
    sections.append(f"\nRaw validation errors:\n{raw_errors}")

    # Field-specific correction guide (only for failed fields)
    sections.append("")
    sections.append("══════════════════════════════════════════")
    sections.append("CORRECTION RULES FOR EACH FAILED FIELD")
    sections.append("══════════════════════════════════════════")
    for field in sorted(fix_targets):
        if field in _FIELD_CORRECTION_GUIDE:
            sections.append(_FIELD_CORRECTION_GUIDE[field])

    # Output instructions
    sections.append("")
    sections.append("══════════════════════════════════════════")
    sections.append("YOUR TASK")
    sections.append("══════════════════════════════════════════")
    sections.append(
        f"Reason step by step about ONLY these fields: {', '.join(sorted(fix_targets))}.\n"
        f"Label each: REASONING FOR [field_name]: ...\n"
        f"Then output a JSON object with ONLY these keys: {', '.join(sorted(fix_targets))}.\n"
        f"No other keys. No markdown fences. Reasoning first, then JSON."
    )

    return "\n".join(sections)


STAGE3_COT_PROMPT = """\
Two previous attempts to extract a FIRAC from this judgment both failed quality checks.
You must now reason more carefully and produce a correct output.

FAILURES FROM PREVIOUS ATTEMPTS:
{defects}

Now follow this procedure STEP BY STEP, being especially careful about the failed fields:

STEP 1 — Read the judgment and list the key events in chronological order.
          Do NOT use legal conclusion words (held, ruled, convicted, directed).

STEP 2 — Write the ISSUE starting with "Whether".
          Check: does it contain any court names (High Court, District Court)?
          If yes, remove them and rewrite abstractly.

STEP 3 — Find the RULE. Write its numbered elements (1), (2), (3).
          Check: does it have at least (1) and (2)? If not, look harder.

STEP 4 — Write the APPLICATION in at least 80 words.
          Connect specific facts to specific rule elements.
          Include the sentence: "The alternative interpretation that [X] fails because [Y]."

STEP 5 — Write the CONCLUSION in 1-2 sentences with an outcome verb.

After your step-by-step reasoning, output the corrected JSON.

JUDGMENT:
{context}

Remember: reasoning first, JSON last. The reasoning you write now directly shapes your JSON answer."""

STAGE3_SYSTEM_PROMPT = """You are a legal reasoning engine that extracts FIRAC structures \
from Indian Supreme Court judgments.

WHAT IS FIRAC? (Read this carefully)

FIRAC stands for:
  F = FACTS       — What actually happened? Who did what to whom?
  I = ISSUE       — What is the abstract legal question the court had to decide?
  R = RULE        — What is the legal test or doctrine the court applied?
  A = APPLICATION — How did the court apply that rule to these specific facts?
  C = CONCLUSION  — What was the final order of the court?

Think of FIRAC like this:
  - FACTS  = the story
  - ISSUE  = the question the story raises
  - RULE   = the law that answers that kind of question
  - APPLICATION = connecting the story to the law, step by step
  - CONCLUSION = the verdict

For this attempt you MUST write step-by-step reasoning BEFORE the JSON.
Write your reasoning labeled STEP 1 through STEP 5, then output the raw JSON after.
No markdown fences around the JSON. The reasoning must come first."""

cot_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", COT_SYSTEM_PROMPT),
        (
            "user",
            """JUDGMENT TEXT:
{context}

─────────────────────────────────────────────
Now follow the 7-step procedure described above.
Write your reasoning in SECTION A first (label each step).
Then write the JSON in SECTION B.
Do not skip the reasoning — it must come before the JSON.""",
        ),
    ]
)
