from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from text_extract import *
from models import *
from prompts import *
from sanitizers import _postprocess_firac
from validation import *
from utils import _extract_partial_json, _extract_cot_trace

TEACHER_MODEL = "phi4:14b"
_BASE_CTX = 16000

llm_teacher_raw = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=True,
    num_ctx=_BASE_CTX,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
)

llm_teacher = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=_BASE_CTX,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
).with_structured_output(FIRACFormat)

llm_fix = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=16000,
    temperature=0.1,  # near-deterministic
    top_p=0.90,  # prune low-probability tokens
    top_k=20,
)

llm_retry = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=16000,
    temperature=0.3,  # reduced
    top_p=0.92,  # tighter nucleus
    top_k=25,
).with_structured_output(FIRACFormat)

llm_retry_raw = ChatOllama(
    model=TEACHER_MODEL,
    validate_model_on_init=False,
    num_ctx=_BASE_CTX,
    temperature=0.3,
    top_p=0.92,
    top_k=25,
)


def _raw_invoke(messages, stage: int) -> str:
    """Invoke the base model without structured output to capture full text."""
    try:
        if stage == 1:
            response = llm_teacher_raw.invoke(messages)
        elif stage == 3:
            response = llm_retry_raw.invoke(messages)
        else:
            print(f"[WARN] _raw_invoke called with unsupported stage {stage}")
            return ""
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[WARN] Raw invoke failed: {e}")
        return ""


def process_case(
    context: str, doc_name: str = "unknown"
) -> Optional[DistillationRecord]:
    messages = cot_generation_prompt.format_messages(context=context)

    raw_text_s1 = _raw_invoke(messages, 1)
    response_s1 = llm_teacher.invoke(messages)
    cot_trace_s1 = (
        _extract_cot_trace(raw_text_s1) if raw_text_s1 else "[trace unavailable]"
    )

    # Phase 1: Post-process before validation
    response_s1 = _postprocess_firac(response_s1)
    is_valid, reason, failed_fields = validate_firac(response_s1)

    if is_valid:
        print("[PASS] Stage 1 (CoT greedy) passed")
        return DistillationRecord(
            file=doc_name,
            judgment=context,
            cot_trace=cot_trace_s1,
            firac=response_s1.model_dump(),
            stage=1,
        )

    print("[FAILED RESPONSE]")
    print(response_s1)

    print(f"[Stage 1 FAIL] Fields: {failed_fields}")
    stage1_reason = reason
    stage1_response = response_s1

    FIRAC_FIELDS = {"question", "issue", "facts", "rule", "application", "conclusion"}

    fix_targets = set(failed_fields) & FIRAC_FIELDS
    if "cross" in failed_fields:
        fix_targets.add("application")

    stage1_dict = stage1_response.model_dump()
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

    # Stage 2 raw LLM — model reasons about and outputs only the failed fields
    raw_fix_response = llm_fix.invoke(fix_messages)
    fix_text = (
        raw_fix_response.content
        if hasattr(raw_fix_response, "content")
        else str(raw_fix_response)
    )
    print(f"[Stage 2] Raw fix response length: {len(fix_text)} chars")
    partial_fix = _extract_partial_json(fix_text)

    stage2_reason = stage1_reason  # default: carry Stage 1 errors into Stage 3
    stage2_failed_fields = list(fix_targets)

    if partial_fix is None:
        print("[Stage 2 FAIL] Could not parse partial JSON from fix response")
    else:
        # Merge: start from Stage 1 output, overlay only the fixed fields
        merged = stage1_response.model_dump()
        applied_fixes = []
        for field in fix_targets:
            if field in partial_fix:
                merged[field] = partial_fix[field]
                applied_fixes.append(field)
        if applied_fixes:
            print(f"[Stage 2] Applied fixes to: {', '.join(applied_fixes)}")

        try:
            response_s2 = FIRACFormat(**merged)
        except Exception as e:
            print(f"[Stage 2 FAIL] Merge produced invalid FIRAC: {e}")
            response_s2 = None

        if response_s2 is not None:
            response_s2 = _postprocess_firac(response_s2)
            is_valid, reason_s2, failed_fields_s2 = validate_firac(response_s2)
            if is_valid:
                print("[PASS] Stage 2 (surgical fix) passed")
                return DistillationRecord(
                    file=doc_name,
                    judgment=context,
                    cot_trace=(
                        _extract_cot_trace(fix_text) if fix_text else cot_trace_s1
                    ),
                    firac=response_s2.model_dump(),
                    stage=2,
                )
            stage2_reason = reason_s2
            stage2_failed_fields = failed_fields_s2

    print(f"[Stage 2 FAIL] Fields: {stage2_failed_fields}")

    stage3_user_msg = STAGE3_COT_PROMPT.format(
        defects=stage2_reason,
        context=context[:12000],
    )

    retry_messages = [
        SystemMessage(content=STAGE3_SYSTEM_PROMPT),
        HumanMessage(content=stage3_user_msg),
    ]

    raw_text_s3 = _raw_invoke(retry_messages, 3)
    response_s3 = llm_retry.invoke(retry_messages)

    response_s3 = _postprocess_firac(response_s3)
    is_valid, reason, _ = validate_firac(response_s3)

    if is_valid:
        print("[PASS] Stage 3 (CoT regeneration) passed")
        return DistillationRecord(
            file=doc_name,
            judgment=context,
            cot_trace=(
                _extract_cot_trace(raw_text_s3)
                if raw_text_s3
                else "[trace unavailable]"
            ),
            firac=response_s3.model_dump(),
            stage=3,
        )

    print("[FAILED RESPONSE]")
    print(response_s3)

    print(f"[FINAL DISCARD] All 3 stages failed: {reason}")
    return None


def pipeline(pdf_path: str) -> Optional[DistillationRecord]:
    case = extract_text(pdf_path)
    context = extract_judgement_section(case.content)
    return process_case(context, doc_name=case.doc_name)
