# firac_extraction

A data-generation pipeline that extracts **FIRAC** (Facts · Issue · Rule · Application · Conclusion) structures from Indian Supreme Court judgment PDFs and writes a validated JSONL dataset for fine-tuning a smaller (~4B) legal-reasoning model.

---

## What it does

1. Reads every PDF under a given input directory (recursive).
2. Isolates the judgment section from each document using pattern matching.
3. Guards against oversized inputs — PDFs whose judgment section exceeds 40 000 characters (~10 000 tokens) are skipped before any LLM call.
4. Runs up to **three escalating LLM stages** to produce a validated FIRAC JSON object.
5. Writes each validated record immediately to a JSONL output file.
6. Maintains a **checkpoint** file so interrupted runs resume exactly where they stopped.
7. Prints stage-level metrics after every file and a full statistics block at the end.

---

## Repository layout

```
firac_extraction/
├── config.py          # All tuneable constants (model, context limits, filenames, sampling params)
├── main.py            # Entry point, 3-stage pipeline, batch runner, checkpoint logic
├── models.py          # Pydantic schema (FIRACFormat), DistillationRecord, Case
├── prompts.py         # System prompts, exemplars, stage-2 builder, stage-3 templates
├── sanitizers.py      # Post-processing: strip court names, procedural history, CoT leaks
├── text_extract.py    # pdfplumber extraction + judgment-section isolation
├── utils.py           # _extract_cot_trace(), _extract_partial_json()
├── validation.py      # Per-field validators + cross-validator + soft-pass logic
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
langchain==1.2.15
langchain-core==1.2.28
langchain-ollama==1.1.0
pdfplumber
```

The pipeline uses [Ollama](https://ollama.com) to serve the teacher model locally. Pull the model before running:

```bash
ollama pull phi4:14b
```

---

## Usage

```bash
# Minimal — output goes to firac_dataset.jsonl, checkpoint to checkpoint.json
python main.py /path/to/pdf/directory

# Explicit paths
python main.py /kaggle/input/supreme-court-pdfs \
    --output  /kaggle/working/firac_dataset.jsonl \
    --checkpoint /kaggle/working/checkpoint.json
```

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_dir` | positional | — | Directory searched recursively for `*.pdf` files |
| `--output` | optional | `firac_dataset.jsonl` | Output JSONL file path |
| `--checkpoint` | optional | `checkpoint.json` | Checkpoint JSON file for resume support |

To resume an interrupted run, pass the same `--checkpoint` path. Files already recorded with a terminal status (`pass`, `fail`, `skip_context`, `skip_error`) are replayed into the running metrics and skipped.

---

## Pipeline stages

### Pre-flight check

Before any LLM call, `process_case()` estimates the token count of the judgment section as `len(text) // 4`. If the character count exceeds `MAX_INPUT_CHARS` (default 40 000), the PDF is recorded as `skip_context` and no model invocation occurs.

### Stage 1 — CoT greedy pass

**Model:** `phi4:14b` — temperature `0.0`, top-p `1.0`, top-k `1` (fully deterministic).

Two parallel calls are made: one unstructured (to capture the chain-of-thought trace) and one with `.with_structured_output(FIRACFormat)` (to produce the JSON). The output is post-processed, then validated. A pass at this stage requires zero hard-fail errors.

### Stage 2 — Surgical field fix

**Model:** `phi4:14b` — temperature `0.1`, top-p `0.90`, top-k `20`.

Only the fields that failed Stage 1 validation are sent for correction. The Stage 2 prompt includes: the judgment text (truncated to 12 000 chars), the passing fields as read-only anchors, the exact validation error strings, and per-field correction rules. The model outputs a partial JSON with only the failed keys; these are merged back onto the Stage 1 output and re-validated.

### Stage 3 — Full CoT regeneration

**Model:** `phi4:14b` — temperature `0.3`, top-p `0.92`, top-k `25`.

A full re-extraction from scratch, with the Stage 2 failure reasons injected as explicit defects to avoid. The judgment text is truncated to `STAGE3_CONTEXT_CHARS` (12 000 chars) to fit within the context window. If this stage also fails, the record is discarded.

---

## Validation rules

`validate_firac()` runs seven independent checkers. A record only passes if all hard-fail checks are clear (soft-passable patterns are accepted without retry).

| Field | Key checks |
|---|---|
| `question` | 8–40 words; ≤ 2 non-legal capitalised tokens (guards against party names / case citations) |
| `issue` | Must start with `"Whether"`; must not contain court names or party-role words |
| `facts` | ≥ 40 words; no legal conclusion language; no appellate-journey narrative; no bracket placeholders |
| `rule` | ≥ 25 words; must contain `(1)` and `(2)` or keywords `requires`/`elements`/`test`/`threshold`; no vague generic phrases (`interests of justice`, `depends on facts`, etc.); ≤ 2 past-tense verbs |
| `application` | ≥ 60 words (soft-passable if only this fails); must contain causal language (`because`, `therefore`, `hence`, …); must reject an alternative outcome; ≥ 12 % lexical overlap with `facts` |
| `conclusion` | 8–60 words; must contain an operative outcome verb (`allowed`, `dismissed`, `set aside`, `upheld`, `affirmed`, `remanded`, `quashed`, etc.) |
| `cross` | Rule–application token overlap ≥ 8 %; application must not introduce > 8 capitalised tokens absent from facts |

**Soft pass:** a record is accepted without retry if the only remaining error is `application · too short (N words, minimum 80 preferred)`.

---

## Post-processing (sanitizers)

Applied after JSON parsing in every stage, before validation:

- **`_sanitize_issue` / `_sanitize_question`** — replaces court names (High Court, Supreme Court, Sessions Court, etc.) with `"the court"` and party-role words (appellant, respondent, petitioner, etc.) with `"a party"`.
- **`_sanitize_facts`** — removes sentences whose subject is a court and whose main verb is a legal disposition (`held`, `ruled`, `set aside`, `restored`, `upheld`, `uphold(s/ing)`, `affirm(ed/ing/s)`, `modified`, `reversed`, `dismissed`, `allowed`, `declared`, `confirmed`, `convicted`, `acquitted`, `directed`, `decreed`, `found guilty`, `granted`, `quashed`). Falls back to the original text if fewer than 30 words survive.
- **`_clean_reasoning_leaks`** — strips `STEP N —` headers and `REASONING FOR <field>:` labels that the model sometimes emits inside JSON string values.

---

## Output format

Each line of the JSONL output is a JSON object with this structure:

```json
{
  "file": "2019_SC_123.pdf",
  "judgment": "<full judgment section text>",
  "cot_trace": "<SECTION A chain-of-thought captured from raw model output>",
  "firac": {
    "question": "Whether ...",
    "issue": "Whether ...",
    "facts": "...",
    "rule": "Under Section X ... where (1) ... (2) ...",
    "application": "... because ... therefore ...",
    "conclusion": "The appeal is allowed ..."
  },
  "stage": 1,
  "metadata": {
    "model_name": "phi4:14b",
    "timestamp": "2025-08-14T10:23:45Z",
    "input_token_estimate": 3241
  }
}
```

`stage` records which pipeline stage produced the record (1, 2, or 3). All string values are UTF-8 sanitised before serialisation.

---

## Checkpoint format

`checkpoint.json` maps each PDF filename to a status entry:

```json
{
  "2019_SC_123.pdf": { "status": "pass", "stage": 1 },
  "2020_SC_456.pdf": { "status": "fail" },
  "large_doc.pdf":   { "status": "skip_context" },
  "corrupt.pdf":     { "status": "skip_error" },
  "in_progress.pdf": { "status": "pending" }
}
```

`pending` indicates the process was killed mid-file; those files are reprocessed on the next run. The checkpoint file is written atomically (`.tmp` → `os.replace`) to prevent corruption on crash.

---

## Configuration reference (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `TEACHER_MODEL` | `"phi4:14b"` | Ollama model tag for all three stages |
| `BASE_CTX` | `16000` | Total context-window size in tokens |
| `MAX_INPUT_CHARS` | `40000` | Maximum judgment-section length; PDFs exceeding this are skipped before any LLM call |
| `STAGE3_CONTEXT_CHARS` | `12000` | Judgment text truncation limit for Stage 3 prompt |
| `OUTPUT_FILENAME` | `"firac_dataset.jsonl"` | Default output file |
| `CHECKPOINT_FILENAME` | `"checkpoint.json"` | Default checkpoint file |
| `STAGE1_TEMPERATURE` | `0.0` | Stage 1 sampling temperature (deterministic) |
| `STAGE2_TEMPERATURE` | `0.1` | Stage 2 sampling temperature (near-deterministic) |
| `STAGE3_TEMPERATURE` | `0.3` | Stage 3 sampling temperature (slightly exploratory) |

---

## Design decisions

**Why single-pass inference only?**
The fine-tuning target is a ~4B model that must run under latency constraints. Training on single-pass extractions keeps inference assumptions consistent between the teacher pipeline and the student model.

**Why three stages rather than one?**
High precision is the priority — a bad sample is worse than no sample. Stage 1 catches easy cases cheaply (temperature 0, deterministic). Stage 2 fixes individual fields without throwing away everything the model got right. Stage 3 is a last resort with more temperature to explore a different reasoning path. Records that fail all three are discarded entirely.

**Why no chunking?**
Chunking breaks cross-sentence reasoning that FIRAC analysis requires. The context-length guard (`MAX_INPUT_CHARS`) is the intentional alternative: either the judgment fits in one pass or it is skipped.

**Why skip rather than truncate on context overflow?**
A truncated judgment produces FIRAC fields that cannot be grounded in the full case. The resulting data would train the student model to reason from incomplete evidence, degrading generalisation.
