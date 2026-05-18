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

### Stage 1 — CoT greedy pass

**Model:** `phi4:14b` — temperature `0.0`, top-p `1.0`, top-k `1` (fully deterministic).

Two parallel calls are made: one unstructured (to capture the chain-of-thought trace) and one with `.with_structured_output(FIRACFormat)` (to produce the JSON). The output is post-processed, then validated. A pass at this stage requires zero hard-fail errors.

### Stage 2 — Surgical field fix

**Model:** `phi4:14b` — temperature `0.1`, top-p `0.90`, top-k `20`.

Only the fields that failed Stage 1 validation are sent for correction. The Stage 2 prompt includes: the judgment text (truncated to 12 000 chars), the passing fields as read-only anchors, the exact validation error strings, and per-field correction rules. The model outputs a partial JSON with only the failed keys; these are merged back onto the Stage 1 output and re-validated.

### Stage 3 — Full CoT regeneration

**Model:** `phi4:14b` — temperature `0.3`, top-p `0.92`, top-k `25`.

A full re-extraction from scratch, with the Stage 2 failure reasons injected as explicit defects to avoid. The judgment text is truncated to `STAGE3_CONTEXT_CHARS` (12 000 chars) to fit within the context window. If this stage also fails, the record is discarded.
