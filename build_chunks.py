# -*- coding: utf-8 -*-
"""
Local stand-in for Azure Document Intelligence Layout parsing + the
section/table-aware chunker described in Phase 2-3 of the RAG guide.

This runs entirely offline (pdfplumber) so the corpus can be inspected now;
swap `extract_layout()` for the Document Intelligence call in the guide when
running against real Azure resources (interface is kept 1:1 compatible).

Output: rag_corpus/chunks.jsonl  -- one JSON object per chunk, ready to embed
and upload to Azure AI Search per Phase 4 of the guide.
"""
import pdfplumber, re, json, os

RAW_DIR = "/home/claude/rag_corpus/raw"
OUT_PATH = "/home/claude/rag_corpus/chunks.jsonl"

# ---------------------------------------------------------------------------
# Boilerplate stripping (SPECIMEN banner / watermark / footer noise)
# ---------------------------------------------------------------------------
BANNER_PATTERNS = [
    r"SPECIMEN\s*/\s*SAMPLE DOCUMENT.*?REAL POLICY",
    r"SPECIMEN\s*/\s*SAMPLE DOCUMENT.*?CDI RECORD",
    r"SPECIMEN\s*/\s*SAMPLE POLICY FORM.*?INSURANCE CONTRACT",
    r"SAMPLE\s*[—\-]\s*NOT A BINDING CONTRACT",
    r"SAMPLE\s*[—\-]\s*NOT A REAL REGULATORY FILING",
    r"SAMPLE\s*[—\-]\s*NOT AN ACTUAL POLICY FORM",
    r"SAMPLE\s*[—\-]\s*NOT A REAL RATE MANUAL",
    r"^SPECIMEN$",
    r"Pacific Meridian Mutual Insurance Company.*?Specimen Policy Form CA-PAP-2026",
    r"Specimen (Rate Manual Page|SERFF Filing|Policy Form).*?\(fictional\)",
    r"^Page \d+$",
]
BANNER_RE = re.compile("|".join(BANNER_PATTERNS), re.IGNORECASE | re.DOTALL)

def strip_boilerplate(text: str) -> str:
    text = BANNER_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Section-heading detection heuristic (stand-in for DI paragraph role ==
# "sectionHeading"): our navy section-bar headers are short, upper-case,
# and match a known label vocabulary used across the corpus.
# ---------------------------------------------------------------------------
SECTION_HEADING_RE = re.compile(
    r"^(PART [A-D][^\n]{0,60}|GENERAL DEFINITIONS|CALIFORNIA-REQUIRED NOTICES.*|"
    r"SCHEDULE OF.*|GENERAL PROVISIONS AND CONDITIONS|IN WITNESS WHEREOF|"
    r"LIENHOLDER.*|COVERAGE SCHEDULE.*|RATED DRIVER.*|"
    r"GENERAL FILING INFORMATION|COMPANY CONTACT INFORMATION|FILING FEES.*|"
    r"PROPOSED EFFECTIVE DATES|FILING DESCRIPTION.*|CERTIFICATIONS.*|"
    r"RATE/RULE SCHEDULE.*|FORM SCHEDULE.*|PURPOSE AND SCOPE|"
    r"INDICATED VS\. SELECTED.*|LOSS DEVELOPMENT AND TREND|EXPENSE AND PROFIT.*|"
    r"GOOD DRIVER DISCOUNT.*|WHAT IS CHANGING|WHO IS AFFECTED|WHY THE CHANGE.*|"
    r"SCHEDULE$|ADDITIONAL DEFINITIONS|PROVISIONS$|"
    r"BASE RATES.*|DEDUCTIBLE RELATIVITY.*|REPRESENTATIVE TERRITORY.*|NOTE$|"
    r"DISCOUNT FACTORS.*|ELIGIBILITY.*|BODILY INJURY LIABILITY.*FACTORS|"
    r"PROPERTY DAMAGE LIABILITY.*FACTORS|APPLICATION$|"
    r"ILLUSTRATIVE VEHICLE SYMBOL.*|MODEL YEAR AGE FACTOR|"
    r"PREMIUM CALCULATION|TERRITORY AND DISCOUNT.*|SAFETY APPAREL.*PREMIUM|"
    r"MOTORCYCLE CLASSIFICATION.*|ELIGIBILITY AND LICENSE.*)$",
    re.MULTILINE,
)

DOC_TYPE_MAP = {
    "Full_": "policy_wording", "CA_Personal_Auto_Policy_SPECIMEN": "policy_wording",
    "0[6-9]|1[01]|06|07|08|09|10|11": "endorsement",
    "RR-": "rate_rule",
    "01_NAIC|02_Rate|03_Form|04_Actuarial|05_Explanatory": "filing_admin",
}

def classify_doc_type(fname):
    if fname.startswith("Full_") or fname.startswith("CA_Personal_Auto_Policy"):
        return "policy_wording"
    if fname.startswith("RR-"):
        return "rate_rule"
    if fname.startswith(("06_", "07_", "08_", "09_", "10_", "11_")):
        return "endorsement"
    if fname.startswith(("01_", "02_", "03_", "04_", "05_")):
        return "filing_admin"
    return "other"

VEHICLE_MAP = {
    "CA_Personal_Auto_Policy_SPECIMEN": "2022 Toyota Camry SE (John A. Sample)",
    "Full_01_2024_Honda_CRV": "2024 Honda CR-V EX-L (Robert T. Nguyen)",
    "Full_02_2023_Ford_F150": "2023 Ford F-150 XLT Pickup (Angela M. Reyes)",
    "Full_03_2025_Tesla_Model3": "2025 Tesla Model 3 Long Range (Wei Chen)",
    "Full_04_2019_Honda_Civic_MinimumLimits": "2019 Honda Civic LX (Priya S. Anand)",
    "Full_05_2021_Chevrolet_Suburban": "2021 Chevrolet Suburban LT (Marcus D. Williams)",
    "Full_06_2020_Mazda_Miata": "2020 Mazda MX-5 Miata (Sofia G. Martinez)",
    "Full_07_2023_BMW_530i": "2023 BMW 530i xDrive (David K. Osei)",
    "Full_08_2018_Toyota_RAV4_SeniorDriver": "2018 Toyota RAV4 LE (Linda P. Tran)",
    "Full_09_2022_Kawasaki_Ninja650_Motorcycle": "2022 Kawasaki Ninja 650 (Jamal R. Foster)",
}


def extract_layout(pdf_path):
    """Returns list of (page_number, text, tables:list[list[list[str]]])."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append((i, text, tables))
    return pages


def table_to_markdown(table):
    rows = [r for r in table if any(c not in (None, "") for c in r)]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    header = [str(c or "").strip() for c in header]
    md = "| " + " | ".join(header) + " |\n"
    md += "|" + "|".join(["---"] * len(header)) + "|\n"
    for r in body:
        r = [str(c or "").strip().replace("\n", " ") for c in r]
        md += "| " + " | ".join(r) + " |\n"
    return md


def chunk_document(pdf_path):
    fname = os.path.basename(pdf_path)
    stem = fname.rsplit(".", 1)[0]
    doc_type = classify_doc_type(fname)
    vehicle = VEHICLE_MAP.get(stem)
    pages = extract_layout(pdf_path)

    chunks = []
    current_section = "General"
    current_text = []
    current_page = pages[0][0] if pages else 1
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx
        text = strip_boilerplate("\n".join(current_text))
        if len(text) > 15:
            chunk_idx += 1
            chunks.append({
                "chunk_id": f"{stem}__c{chunk_idx}",
                "document_name": fname,
                "doc_type": doc_type,
                "vehicle": vehicle,
                "section": current_section,
                "page_number": current_page,
                "content_type": "text",
                "content": text,
            })

    for page_num, text, tables in pages:
        clean = strip_boilerplate(text)
        lines = clean.split("\n")
        for line in lines:
            line_stripped = line.strip()
            if SECTION_HEADING_RE.match(line_stripped) and len(line_stripped) < 70:
                flush()
                current_section = line_stripped.title()
                current_text = []
                current_page = page_num
            else:
                if line_stripped:
                    current_text.append(line_stripped)
                    current_page = page_num
        # Tables -> atomic chunks, tagged to the section active on this page
        for t_idx, table in enumerate(tables):
            md = table_to_markdown(table)
            if md.strip():
                chunk_idx += 1
                chunks.append({
                    "chunk_id": f"{stem}__c{chunk_idx}_table{t_idx+1}",
                    "document_name": fname,
                    "doc_type": doc_type,
                    "vehicle": vehicle,
                    "section": current_section,
                    "page_number": page_num,
                    "content_type": "table",
                    "content": md,
                })
    flush()
    return chunks


def main():
    all_chunks = []
    files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf"))
    for f in files:
        path = os.path.join(RAW_DIR, f)
        doc_chunks = chunk_document(path)
        all_chunks.extend(doc_chunks)
        print(f"{f}: {len(doc_chunks)} chunks")

    with open(OUT_PATH, "w") as out:
        for c in all_chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nTotal documents: {len(files)}")
    print(f"Total chunks:    {len(all_chunks)}")
    print(f"Written to:      {OUT_PATH}")

    # quick sanity stats
    by_type = {}
    for c in all_chunks:
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1
    print("Chunks by doc_type:", by_type)


if __name__ == "__main__":
    main()
