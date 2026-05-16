import os
import re
import pdfplumber
from models import Case


def extract_text(pdf_path):
    full_text = ""
    doc_name = os.path.basename(pdf_path)
    page_count = 0
    failed_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for i, pages in enumerate(pdf.pages):
            cropped = pages.crop((0, 50, pages.width, pages.height - 40))
            text = cropped.extract_text(x_tolerance=3, y_tolerance=3, layout=True)
            if text:
                full_text += text + "\n"
            else:
                failed_pages.append(i + 1)
    if failed_pages:
        print(f"[WARN] Failed pages: {failed_pages}")
    return Case(doc_name, page_count, full_text)


JUDGMENT_PATTERNS = [
    r"\bJUDGMENT\s*/\s*ORDER\s+OF\s+THE\s+SUPREME\s+COURT\b",
    r"\bJ\s*U\s*D\s*G\s*M\s*E\s*N\s*T\b",
    r"\bO\s*R\s*D\s*E\s*R\b",
    r"\bHON['']\s*BLE\s+MR\.?\s+JUSTICE\b",
    r"\bPER\s+CURIAM\b",
    r"^\s*\d+\.\s+(?:The\s+)?(?:facts|background)",
]


def extract_judgement_section(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    start_idx = None
    for pattern in JUDGMENT_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            if start_idx is None or match.start() < start_idx:
                start_idx = match.start()
    if start_idx is None:
        return text.strip()
    return text[start_idx:].strip()
