import os
import re
import pdfplumber
from models import Case


def extract_text(pdf_path: str) -> Case:
    full_text = ""
    doc_name = os.path.basename(pdf_path)
    failed_pages: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            try:
                cropped = page.crop((0, 50, page.width, page.height - 40))
                text = cropped.extract_text(x_tolerance=3, y_tolerance=3, layout=True)
                if text:
                    full_text += text + "\n"
                else:
                    failed_pages.append(i + 1)
            except Exception as e:
                print(f"[WARN] {doc_name}: page {i + 1} extraction failed — {e}")
                failed_pages.append(i + 1)

    if failed_pages:
        print(f"[WARN] {doc_name}: {len(failed_pages)} page(s) failed: {failed_pages}")

    return Case(doc_name, page_count, full_text)


JUDGMENT_PATTERNS = [
    r"\bJUDGMENT\s*/\s*ORDER\s+OF\s+THE\s+SUPREME\s+COURT\b",
    r"\bJ\s*U\s*D\s*G\s*M\s*E\s*N\s*T\b",
    r"\bO\s*R\s*D\s*E\s*R\b",
    r"\bHON['']?\s*BLE\s+MR\.?\s+JUSTICE\b",
    r"\bPER\s+CURIAM\b",
    r"^\s*\d+\.\s+(?:The\s+)?(?:facts|background)",
]


def extract_judgement_section(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    start_idx = None
    for pattern in JUDGMENT_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            if start_idx is None or match.start() < start_idx:
                start_idx = match.start()
    if start_idx is None:
        return text.strip()
    return text[start_idx:].strip()
