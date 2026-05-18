TEACHER_MODEL: str = "phi4:14b"

BASE_CTX: int = 16_000

MAX_INPUT_CHARS: int = 40_000

STAGE3_CONTEXT_CHARS: int = 12_000

OUTPUT_FILENAME: str = "firac_dataset.jsonl"

CHECKPOINT_FILENAME: str = "checkpoint.json"

STAGE1_TEMPERATURE: float = 0.0
STAGE1_TOP_P: float = 1.0
STAGE1_TOP_K: int = 1

STAGE2_TEMPERATURE: float = 0.1
STAGE2_TOP_P: float = 0.90
STAGE2_TOP_K: int = 20

STAGE3_TEMPERATURE: float = 0.3
STAGE3_TOP_P: float = 0.92
STAGE3_TOP_K: int = 25
