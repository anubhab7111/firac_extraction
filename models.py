from pydantic import BaseModel, Field


class FIRACFormat(BaseModel):
    question: str = Field(
        ..., description="Abstract legal question, no party names or dates"
    )
    issue: str = Field(..., description="Begins with 'Whether', legal principle only")
    facts: str = Field(..., description="Chronological narrative, no legal conclusions")
    rule: str = Field(
        ..., description="Named doctrine with numbered elements/threshold"
    )
    application: str = Field(
        ..., description="Causal reasoning: fact + fact → rule element → outcome"
    )
    conclusion: str = Field(..., description="Operative order only, 1-2 sentences")


class DistillationRecord:
    def __init__(
        self, file: str, judgment: str, cot_trace: str, firac: dict, stage: int
    ):
        self.file = file
        self.judgment = judgment
        self.cot_trace = cot_trace
        self.firac = firac
        self.stage = stage

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "judgment": self.judgment,
            "cot_trace": self.cot_trace,
            "firac": self.firac,
            "stage": self.stage,
        }


class Case:
    def __init__(
        self,
        doc_name: str,
        pages: int,
        content: str,
        firac: dict | None = None,
        firac_time: float | None = None,
    ):
        self.doc_name = doc_name
        self.content = content
        self.pages = pages
        self.firac = firac
        self.firac_time = firac_time
