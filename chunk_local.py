import pdfplumber, re, json, os

RAW_DIR = "./raw_policies"
OUT_PATH = "./chunks.jsonl"

SECTION_HEADING_RE = re.compile(
    r"^(PART [A-D][^\n]{0,60}|GENERAL DEFINITIONS|CALIFORNIA-REQUIRED NOTICES.*|"
    r"SCHEDULE OF.*|GENERAL PROVISIONS AND CONDITIONS|IN WITNESS WHEREOF|"
    r"LIENHOLDER.*|COVERAGE SCHEDULE.*|RATED DRIVER.*)$",
    re.MULTILINE,
)

def extract_layout(pdf_path):
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
    pages = extract_layout(pdf_path)
    chunks, current_section, current_text, current_page, idx = [], "General", [], 1, 0

    def flush():
        nonlocal idx
        text = "\n".join(current_text).strip()
        if len(text) > 15:
            idx += 1
            chunks.append({"chunk_id": f"{stem}__c{idx}", "document_name": fname,
                            "section": current_section, "page_number": current_page,
                            "content_type": "text", "content": text})

    for page_num, text, tables in pages:
        for line in text.split("\n"):
            line = line.strip()
            if SECTION_HEADING_RE.match(line) and len(line) < 70:
                flush(); current_section, current_text = line.title(), []
            elif line:
                current_text.append(line); current_page = page_num
        for t_idx, table in enumerate(tables):
            md = table_to_markdown(table)
            if md.strip():
                idx += 1
                chunks.append({"chunk_id": f"{stem}__c{idx}_table{t_idx+1}", "document_name": fname,
                                "section": current_section, "page_number": page_num,
                                "content_type": "table", "content": md})
    flush()
    return chunks

all_chunks = []
for f in sorted(os.listdir(RAW_DIR)):
    if f.lower().endswith(".pdf"):
        doc_chunks = chunk_document(os.path.join(RAW_DIR, f))
        all_chunks.extend(doc_chunks)
        print(f"{f}: {len(doc_chunks)} chunks")

with open(OUT_PATH, "w") as out:
    for c in all_chunks:
        out.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"\nTotal chunks: {len(all_chunks)} → written to {OUT_PATH}")
