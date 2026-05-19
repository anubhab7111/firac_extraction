# models.py
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional


class FIRACFormat(BaseModel):
    question: str = Field(
        ...,
        description="Abstract legal question, no party names or dates",
    )
    issue: str = Field(
        ...,
        description="Begins with 'Whether', legal principle only",
    )
    facts: str = Field(
        ...,
        description="Chronological narrative, no legal conclusions",
    )
    rule: str = Field(
        ...,
        description="Named doctrine with numbered elements/threshold",
    )
    application: str = Field(
        ...,
        description="Causal reasoning: fact + fact → rule element → outcome",
    )
    conclusion: str = Field(
        ...,
        description="Operative order only, 1-2 sentences",
    )


class DistillationRecord:
    def __init__(
        self,
        file: str,
        judgment: str,
        # cot_trace: str,
        firac: dict,
        stage: int,
        model_name: str = "",
        input_token_estimate: int = 0,
    ):
        self.file = file
        self.judgment = judgment
        # self.cot_trace = cot_trace
        self.firac = firac
        self.stage = stage
        self.model_name = model_name
        self.input_token_estimate = input_token_estimate
        self.timestamp: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _sanitize(value) -> str:
        if not isinstance(value, str):
            return value
        return value.encode("utf-8", errors="replace").decode("utf-8")

    def to_dict(self) -> dict:
        sanitized_firac = {k: self._sanitize(v) for k, v in self.firac.items()}
        return {
            "file": self._sanitize(self.file),
            "judgment": self._sanitize(self.judgment),
            # "cot_trace": self._sanitize(self.cot_trace),
            "firac": sanitized_firac,
            "stage": self.stage,
            "metadata": {
                "model_name": self._sanitize(self.model_name),
                "timestamp": self.timestamp,
                "input_token_estimate": self.input_token_estimate,
            },
        }


class Case:
    def __init__(
        self,
        doc_name: str,
        pages: int,
        content: str,
        firac: Optional[dict] = None,
        firac_time: Optional[float] = None,
    ):
        self.doc_name = doc_name
        self.content = content
        self.pages = pages
        self.firac = firac
        self.firac_time = firac_time
