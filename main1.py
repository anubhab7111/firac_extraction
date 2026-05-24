import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from config import (
    TEACHER_MODEL,
    BASE_CTX,
    MIN_OUTPUT_TOKENS,
    STAGE3_CONTEXT_CHARS,
    OUTPUT_FILENAME,
    CHECKPOINT_FILENAME,
    STAGE1_TEMPERATURE,
    STAGE1_TOP_P,
    STAGE1_TOP_K,
    STAGE2_TEMPERATURE,
    STAGE2_TOP_P,
    STAGE2_TOP_K,
    STAGE3_TEMPERATURE,
    STAGE3_TOP_P,
    STAGE3_TOP_K,
)
from models import DistillationRecord, FIRACFormat
from prompts import (
    cot_generation_prompt,
    STAGE2_SYSTEM_PROMPT,
    STAGE3_SYSTEM_PROMPT,
    STAGE3_COT_PROMPT,
    _build_stage2_prompt,
)
from sanitizers import _postprocess_firac
from text_extract import extract_text, extract_judgement_section
from utils import _extract_partial_json, _extract_cot_trace
from validation import validate_firac

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP_CTX = "skip_context"
STATUS_SKIP_ERROR = "skip_error"

llm_teacher_raw = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=True,
    num_ctx=BASE_CTX,
    temperature=STAGE1_TEMPERATURE,
    top_p=STAGE1_TOP_P,
    top_k=STAGE1_TOP_K,
)

llm_teacher = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=BASE_CTX,
    temperature=STAGE1_TEMPERATURE,
    top_p=STAGE1_TOP_P,
    top_k=STAGE1_TOP_K,
).with_structured_output(FIRACFormat)

llm_fix = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=BASE_CTX,
    temperature=STAGE2_TEMPERATURE,
    top_p=STAGE2_TOP_P,
    top_k=STAGE2_TOP_K,
)

llm_retry = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=BASE_CTX,
    temperature=STAGE3_TEMPERATURE,
    top_p=STAGE3_TOP_P,
    top_k=STAGE3_TOP_K,
).with_structured_output(FIRACFormat)

llm_retry_raw = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=BASE_CTX,
    temperature=STAGE3_TEMPERATURE,
    top_p=STAGE3_TOP_P,
    top_k=STAGE3_TOP_K,
)


# def _raw_invoke(messages, stage: int) -> str:
#     try:
#         if stage == 1:
#             response = llm_teacher_raw.invoke(messages)
#         elif stage == 3:
#             response = llm_retry_raw.invoke(messages)
#         else:
#             print(f"[WARN] _raw_invoke called with unsupported stage {stage}")
#             return ""
#         return response.content if hasattr(response, "content") else str(response)
#     except Exception as e:
#         print(f"[WARN] Raw invoke failed at stage {stage}: {e}")
#         return ""


def _empty_metrics() -> dict:
    return {
        "text_extraction_failed": 0,
        "skipped_context_too_long": 0,
        "stage1_pass": 0,
        "stage2_pass": 0,
        "stage3_pass": 0,
        "discarded": 0,
    }


def _print_metrics(
    metrics: dict,
    processed: int,
    total: int,
    elapsed: float,
    *,
    final: bool = False,
) -> None:
    passed = metrics["stage1_pass"] + metrics["stage2_pass"] + metrics["stage3_pass"]
    attempted = passed + metrics["discarded"]
    pass_rate = (passed / attempted * 100) if attempted > 0 else 0.0

    border = "═" * 58 if final else "─" * 58
    label = "  FINAL STATISTICS" if final else "  RUNNING METRICS"

    print(f"\n{border}")
    print(label)
    print(border)
    print(f"  PDFs processed         : {processed} / {total}")
    print(f"  Text extraction failed : {metrics['text_extraction_failed']}")
    print(f"  Skipped (ctx too long) : {metrics['skipped_context_too_long']}")
    print(f"  LLM attempted          : {attempted}")
    print(f"    Stage 1 pass         : {metrics['stage1_pass']}", end="")
    if attempted:
        print(f"  ({metrics['stage1_pass']/attempted*100:.1f} %)", end="")
    print()
    print(f"    Stage 2 pass         : {metrics['stage2_pass']}", end="")
    if attempted:
        print(f"  ({metrics['stage2_pass']/attempted*100:.1f} %)", end="")
    print()
    print(f"    Stage 3 pass         : {metrics['stage3_pass']}", end="")
    if attempted:
        print(f"  ({metrics['stage3_pass']/attempted*100:.1f} %)", end="")
    print()
    print(f"  Discarded (all fail)   : {metrics['discarded']}")
    print(f"  Overall pass rate      : {pass_rate:.1f} %")
    print(f"  Elapsed                : {elapsed:.0f}s")

    if final and passed > 0:
        avg_stages = (
            metrics["stage1_pass"] * 1
            + metrics["stage2_pass"] * 2
            + metrics["stage3_pass"] * 3
        ) / passed
        print(f"  Avg stages per record  : {avg_stages:.2f}")

    print(f"{border}\n")


def process_case(
    context: str,
    doc_name: str,
    metrics: dict,
) -> tuple[Optional[DistillationRecord], str]:
    messages = cot_generation_prompt.format_messages(context=context)
    # prompt_text = "\n".join(f"{m.type}: {m.content}" for m in messages)
    # prompt_tokens = llm_teacher_raw.get_num_tokens(prompt_text)
    # token_budget = BASE_CTX - MIN_OUTPUT_TOKENS
    #
    # if prompt_tokens > token_budget:
    #     print(
    #         f"[SKIP] {doc_name}: full prompt is {prompt_tokens:,} tokens "
    #         f"(budget {token_budget:,} = {BASE_CTX:,} ctx − {MIN_OUTPUT_TOKENS:,} output reserve). "
    #         f"Skipping."
    #     )
    #     metrics["skipped_context_too_long"] += 1
    #     return None, STATUS_SKIP_CTX

    # raw_text_s1 = _raw_invoke(messages, 1)
    response_s1 = llm_teacher.invoke(messages)
    # cot_trace_s1 = (
    #     _extract_cot_trace(raw_text_s1) if raw_text_s1 else "[trace unavailable]"
    # )

    response_s1 = _postprocess_firac(response_s1)
    is_valid, reason, failed_fields = validate_firac(response_s1)

    if is_valid:
        print(f"[PASS] {doc_name}: Stage 1")
        metrics["stage1_pass"] += 1
        return (
            DistillationRecord(
                file=doc_name,
                judgment=context,
                # cot_trace=cot_trace_s1,
                firac=response_s1.model_dump(),
                stage=1,
                model_name=TEACHER_MODEL,
                # input_token_estimate=prompt_tokens,
            ),
            STATUS_PASS,
        )

    print(f"[FAIL] {doc_name}: Stage 1 — fields: {failed_fields}")

    FIRAC_FIELDS = {"question", "issue", "facts", "rule", "application", "conclusion"}
    fix_targets = set(failed_fields) & FIRAC_FIELDS
    if "cross" in failed_fields:
        fix_targets.add("application")

    stage1_dict = response_s1.model_dump()
    fix_message = _build_stage2_prompt(
        context=context,
        stage1_firac=stage1_dict,
        fix_targets=fix_targets,
        raw_errors=reason,
    )
    fix_messages = [
        SystemMessage(content=STAGE2_SYSTEM_PROMPT),
        HumanMessage(content=fix_message),
    ]

    raw_fix_response = llm_fix.invoke(fix_messages)
    fix_text = (
        raw_fix_response.content
        if hasattr(raw_fix_response, "content")
        else str(raw_fix_response)
    )
    print(f"[Stage 2] {doc_name}: raw fix response {len(fix_text):,} chars")
    partial_fix = _extract_partial_json(fix_text)

    stage2_reason = reason
    stage2_failed_fields = list(fix_targets)

    if partial_fix is None:
        print(
            f"[FAIL] {doc_name}: Stage 2 — could not parse partial JSON from fix response"
        )
    else:
        merged = stage1_dict.copy()
        applied_fixes: list[str] = []
        for field in fix_targets:
            if field in partial_fix:
                merged[field] = partial_fix[field]
                applied_fixes.append(field)
        if applied_fixes:
            print(f"[Stage 2] {doc_name}: applied fixes to {', '.join(applied_fixes)}")

        try:
            response_s2 = FIRACFormat(**merged)
        except Exception as e:
            print(f"[FAIL] {doc_name}: Stage 2 — merge produced invalid FIRAC: {e}")
            response_s2 = None

        if response_s2 is not None:
            response_s2 = _postprocess_firac(response_s2)
            is_valid, reason_s2, failed_fields_s2 = validate_firac(response_s2)
            if is_valid:
                print(f"[PASS] {doc_name}: Stage 2")
                metrics["stage2_pass"] += 1
                return (
                    DistillationRecord(
                        file=doc_name,
                        judgment=context,
                        # cot_trace=(
                        #     _extract_cot_trace(fix_text) if fix_text else cot_trace_s1
                        # ),
                        firac=response_s2.model_dump(),
                        stage=2,
                        model_name=TEACHER_MODEL,
                        # input_token_estimate=prompt_tokens,
                    ),
                    STATUS_PASS,
                )
            stage2_reason = reason_s2
            stage2_failed_fields = failed_fields_s2

    print(f"[FAIL] {doc_name}: Stage 2 — fields: {stage2_failed_fields}")

    stage3_user_msg = STAGE3_COT_PROMPT.format(
        defects=stage2_reason,
        context=context[:STAGE3_CONTEXT_CHARS],
    )
    retry_messages = [
        SystemMessage(content=STAGE3_SYSTEM_PROMPT),
        HumanMessage(content=stage3_user_msg),
    ]

    # raw_text_s3 = _raw_invoke(retry_messages, 3)
    response_s3 = llm_retry.invoke(retry_messages)
    response_s3 = _postprocess_firac(response_s3)
    is_valid, reason, _ = validate_firac(response_s3)

    if is_valid:
        print(f"[PASS] {doc_name}: Stage 3")
        metrics["stage3_pass"] += 1
        return (
            DistillationRecord(
                file=doc_name,
                judgment=context,
                # cot_trace=(
                #     _extract_cot_trace(raw_text_s3)
                #     if raw_text_s3
                #     else "[trace unavailable]"
                # ),
                firac=response_s3.model_dump(),
                stage=3,
                model_name=TEACHER_MODEL,
                # input_token_estimate=prompt_tokens,
            ),
            STATUS_PASS,
        )

    print(f"[DISCARD] {doc_name}: all 3 stages failed — {reason}")
    metrics["discarded"] += 1
    return None, STATUS_FAIL


def pipeline(
    pdf_path: str,
    metrics: dict,
) -> tuple[Optional[DistillationRecord], str]:
    try:
        case = extract_text(pdf_path)
    except Exception as e:
        print(f"[ERROR] {os.path.basename(pdf_path)}: text extraction failed — {e}")
        metrics["text_extraction_failed"] += 1
        return None, STATUS_SKIP_ERROR

    context = extract_judgement_section(case.content)
    return process_case(context, doc_name=case.doc_name, metrics=metrics)


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"[INFO] Loaded checkpoint from {path} ({len(data)} entries)")
            return data
        except Exception as e:
            print(f"[WARN] Could not load checkpoint ({e}) — starting fresh")
    return {}


def _save_checkpoint(checkpoint: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _append_record(record: DistillationRecord, output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def run(input_dir: str, output_path: str, checkpoint_path: str) -> None:
    pdf_files = sorted(Path(input_dir).rglob("*.pdf"))
    total = len(pdf_files)
    if total == 0:
        print(f"[ERROR] No PDF files found under {input_dir}")
        sys.exit(1)

    print(f"[INFO] Found {total} PDF(s) under {input_dir}")
    print(f"[INFO] Output  : {output_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    # print(f"[INFO] Max input chars: {MAX_INPUT_CHARS:,}\n")

    checkpoint = _load_checkpoint(checkpoint_path)
    metrics = _empty_metrics()
    start_time = time.time()
    processed = 0

    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        ck_entry = checkpoint.get(pdf_name, {})
        ck_status = ck_entry.get("status")

        if ck_status in (STATUS_PASS, STATUS_FAIL, STATUS_SKIP_CTX, STATUS_SKIP_ERROR):
            if ck_status == STATUS_PASS:
                stage = ck_entry.get("stage", 1)
                metrics[f"stage{stage}_pass"] += 1
            elif ck_status == STATUS_FAIL:
                metrics["discarded"] += 1
            elif ck_status == STATUS_SKIP_CTX:
                metrics["skipped_context_too_long"] += 1
            elif ck_status == STATUS_SKIP_ERROR:
                metrics["text_extraction_failed"] += 1
            processed += 1
            continue

        print(f"\n[{processed + 1}/{total}] {pdf_name}")
        checkpoint[pdf_name] = {"status": "pending"}
        _save_checkpoint(checkpoint, checkpoint_path)

        record, status = pipeline(str(pdf_path), metrics)
        processed += 1

        if record is not None:
            _append_record(record, output_path)
            checkpoint[pdf_name] = {"status": STATUS_PASS, "stage": record.stage}
            print(f"[SAVED] {pdf_name} → stage {record.stage}")
        else:
            checkpoint[pdf_name] = {"status": status}

        _save_checkpoint(checkpoint, checkpoint_path)

        elapsed = time.time() - start_time
        _print_metrics(metrics, processed, total, elapsed)

    elapsed = time.time() - start_time
    _print_metrics(metrics, processed, total, elapsed, final=True)

    passed = metrics["stage1_pass"] + metrics["stage2_pass"] + metrics["stage3_pass"]
    print(f"  Records written to : {output_path}")
    print(f"  Total records saved: {passed}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract FIRAC structures from Indian Supreme Court judgment PDFs ",
        ),
    )
    parser.add_argument(
        "input_dir",
        default="./data/",
        help="Directory containing PDF files (searched recursively for *.pdf)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILENAME,
        metavar="PATH",
        help=f"Output JSONL file  (default: {OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_FILENAME,
        metavar="PATH",
        help=f"Checkpoint JSON file for resume support  (default: {CHECKPOINT_FILENAME})",
    )
    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
