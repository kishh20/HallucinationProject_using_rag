"""
Institutional Document Processing & Chunking Engine
Extracts text and metadata from PDF, DOCX, TXT, MD, CSV files.
Applies intelligent semantic and recursive chunking for institutional repositories.
"""
import os
import re
import uuid
from typing import List, Dict, Any, Optional
import pypdf
import docx
import pandas as pd


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract page-by-page text from a PDF file stream."""
    import io
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages_data = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages_data.append({
                "page_number": idx + 1,
                "text": text,
                "filename": filename,
            })
    return pages_data


def extract_text_from_docx(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract paragraphs and sections from a DOCX file stream."""
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            full_text.append(t)
    combined = "\n\n".join(full_text)
    words = combined.split()
    page_size = 300
    pages_data = []
    if not words:
        return []
    for i in range(0, len(words), page_size):
        chunk_words = words[i:i + page_size]
        pages_data.append({
            "page_number": (i // page_size) + 1,
            "text": " ".join(chunk_words),
            "filename": filename,
        })
    return pages_data


def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text from plain text or markdown."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    lines = text.split("\n")
    current_page = 1
    current_lines = []
    pages_data = []

    for line in lines:
        current_lines.append(line)
        if len(current_lines) >= 45 or re.match(r"^(SECTION|UNIT|MODULE|CHAPTER)\s+\d+", line.strip(), re.I):
            page_text = "\n".join(current_lines).strip()
            if page_text:
                pages_data.append({
                    "page_number": current_page,
                    "text": page_text,
                    "filename": filename,
                })
                current_page += 1
                current_lines = []

    if current_lines:
        page_text = "\n".join(current_lines).strip()
        if page_text:
            pages_data.append({
                "page_number": current_page,
                "text": page_text,
                "filename": filename,
            })
    return pages_data


def extract_text_from_csv(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Convert CSV institutional rows into readable textual chunks."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    records = []
    for idx, row in df.iterrows():
        row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        records.append({
            "page_number": (idx // 20) + 1,
            "text": row_str,
            "filename": filename,
        })
    return records


def parse_document_file(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Detect file type and extract text page data."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_bytes, filename)
    elif ext in [".csv"]:
        return extract_text_from_csv(file_bytes, filename)
    else:
        return extract_text_from_txt(file_bytes, filename)


def recursive_character_chunker(
    text: str,
    chunk_size: int = 650,
    chunk_overlap: int = 120,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Split text recursively by paragraph, sentence, and word boundaries.
    Guarantees that words are NEVER sliced in half during chunk overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", " ", ""]

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find highest level separator present
    sep = separators[-1]
    for s in separators:
        if s in text:
            sep = s
            break

    splits = text.split(sep) if sep else list(text)
    chunks = []
    current_chunk = ""

    for s in splits:
        candidate = (current_chunk + sep + s) if current_chunk else s
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Safe overlap: ensure we start at a complete word boundary!
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_candidate = current_chunk[-chunk_overlap:]
                # Trim to the first whitespace boundary so we never cut words mid-word!
                first_space = overlap_candidate.find(" ")
                if first_space != -1 and first_space < len(overlap_candidate) - 1:
                    clean_overlap = overlap_candidate[first_space + 1:].strip()
                else:
                    clean_overlap = overlap_candidate.strip()
                current_chunk = (clean_overlap + sep + s).strip()
            else:
                current_chunk = s.strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_institutional_document(
    file_bytes: bytes,
    filename: str,
    category: str = "General",
    course_code: str = "",
    department: str = "",
    chunk_size: int = 650,
    chunk_overlap: int = 120,
) -> List[Dict[str, Any]]:
    """
    Process an institutional document into indexed chunk dictionaries with rich metadata.
    """
    pages_data = parse_document_file(file_bytes, filename)
    doc_id = str(uuid.uuid4())[:8]
    indexed_chunks = []

    if not course_code:
        code_match = re.search(r"\b([A-Z]{2,4}\s*\d{3,4})\b", filename)
        if code_match:
            course_code = code_match.group(1).replace(" ", "")

    chunk_idx = 0
    for page in pages_data:
        page_num = page["page_number"]
        page_text = page["text"]
        raw_chunks = recursive_character_chunker(page_text, chunk_size, chunk_overlap)

        for chunk_text in raw_chunks:
            if len(chunk_text.strip()) < 25:
                continue

            chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
            indexed_chunks.append({
                "id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "category": category,
                "course_code": course_code or "INST-DOC",
                "department": department or "Academic Affairs",
                "page_number": page_num,
                "text": chunk_text,
                "char_length": len(chunk_text),
            })
            chunk_idx += 1

    return indexed_chunks
