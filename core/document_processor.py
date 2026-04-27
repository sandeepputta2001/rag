"""
Section 2 – Data Ingestion & Storage Baseline

Implements:
  • Multi-format parsing
      – PDF with pdfplumber (text + table extraction); pytesseract OCR fallback
      – HTML with BeautifulSoup4 (removes noise tags, preserves heading structure)
      – Markdown with header-based splitting (H1/H2/H3)
      – Python source files with AST-based top-level chunking
  • Chunking strategies
      – Fixed character chunking with configurable overlap
      – Fixed token chunking via tiktoken (falls back to char-based estimate)
      – Recursive chunking with separator-priority list (\n\n → \n → ". " → " " → "")
  • Rich metadata schema
      source, doc_type, page_number, section, ingested_at,
      access_level, chunk_index, total_chunks, token_count
  • Content-hash chunk IDs — idempotent re-indexing (same content = same ID)
  • Traceability logging per document and per chunk
"""

from __future__ import annotations

import re
import ast as _ast
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.foundations import count_tokens, make_chunk_id

DOCS_DIR = Path(__file__).parent.parent / "techcorp-docs"

# ---------------------------------------------------------------------------
# Metadata schema helpers
# ---------------------------------------------------------------------------

def build_metadata(
    source: str,
    doc_type: str,
    chunk_index: int,
    total_chunks: int,
    page_number: Optional[int] = None,
    section: Optional[str] = None,
    access_level: str = "public",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical metadata schema used for every chunk across all document types.
    Adding page_number/section only when relevant keeps ChromaDB metadata compact.
    """
    meta: Dict[str, Any] = {
        "source": source,
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "access_level": access_level,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    if page_number is not None:
        meta["page_number"] = page_number
    if section is not None:
        meta["section"] = section
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# PDF Parsing (pdfplumber + OCR fallback)
# ---------------------------------------------------------------------------

def parse_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Extract text and tables from a PDF using pdfplumber.
    Tables are converted to pipe-delimited markdown so they survive chunking.
    Returns list of {"page": int, "text": str, "tables": list[str]}.
    """
    try:
        import pdfplumber  # type: ignore
        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                raw_tables = page.extract_tables() or []
                table_md: List[str] = []
                for tbl in raw_tables:
                    if tbl:
                        rows = [
                            " | ".join("" if cell is None else str(cell) for cell in row)
                            for row in tbl
                        ]
                        table_md.append("\n".join(rows))
                pages.append({"page": i + 1, "text": text, "tables": table_md})
        return pages
    except ImportError:
        # pdfplumber not installed — minimal fallback via pdfminer
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            text = extract_text(path) or ""
            return [{"page": 1, "text": text, "tables": []}]
        except Exception:
            return [{"page": 1, "text": "", "tables": []}]


def parse_pdf_with_ocr(path: str) -> List[Dict[str, Any]]:
    """
    OCR path for scanned / image-only PDFs.
    Requires pytesseract + pdf2image (poppler).  Falls back to pdfplumber.
    """
    try:
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
        images = convert_from_path(path)
        return [
            {"page": i + 1, "text": pytesseract.image_to_string(img), "tables": []}
            for i, img in enumerate(images)
        ]
    except ImportError:
        return parse_pdf(path)


# ---------------------------------------------------------------------------
# HTML Parsing (BeautifulSoup4)
# ---------------------------------------------------------------------------

_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]


def parse_html(html: str, url: str = "") -> str:
    """
    Extract clean prose from HTML.
    Removes noise tags (nav, footer, script…) and preserves heading structure
    by converting <h1>–<h4> to Markdown ## markers before stripping all tags.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        # Rewrite headings to Markdown so downstream chunkers can split on them
        for level in range(1, 5):
            for h in soup.find_all(f"h{level}"):
                prefix = "#" * level
                h.replace_with(f"\n\n{prefix} {h.get_text(strip=True)}\n\n")
        # Preserve paragraphs as blank-line separated blocks
        for p in soup.find_all("p"):
            p.replace_with(f"\n\n{p.get_text(strip=True)}\n\n")
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # Regex fallback when BeautifulSoup is absent
        return re.sub(r"<[^>]+>", " ", html).strip()


# ---------------------------------------------------------------------------
# Markdown Parsing — header-based section splitting
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def parse_markdown_by_headers(text: str) -> List[Dict[str, Any]]:
    """
    Split Markdown text at H1/H2/H3 boundaries.
    Returns list of {"section": str, "level": int, "content": str}.
    Each section's content does NOT include its own heading line.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [{"section": "main", "level": 0, "content": text}]

    sections: List[Dict[str, Any]] = []

    # Text before the first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"section": "preamble", "level": 0, "content": preamble})

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({
            "section": m.group(2).strip(),
            "level": len(m.group(1)),
            "content": content,
        })

    return [s for s in sections if s["content"]]


# ---------------------------------------------------------------------------
# Code Chunking — AST-aware Python; regex for other languages
# ---------------------------------------------------------------------------

def chunk_code(code: str, language: str = "python", max_chars: int = 1_500) -> List[str]:
    """
    Split source code into meaningful chunks.
    For Python: uses the AST to extract top-level functions and classes.
    For other languages: falls back to blank-line block splitting.
    """
    if language == "python":
        try:
            tree = _ast.parse(code)
            lines = code.splitlines(keepends=True)
            chunks: List[str] = []
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                    if node.col_offset == 0:  # top-level only
                        chunk = "".join(lines[node.lineno - 1 : node.end_lineno])
                        if chunk.strip():
                            chunks.append(chunk)
            if chunks:
                return chunks
        except SyntaxError:
            pass

    # Generic: split on blank lines then merge small adjacent blocks
    blocks = re.split(r"\n{2,}", code)
    merged: List[str] = []
    current = ""
    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = block if len(block) <= max_chars else block[:max_chars]
    if current:
        merged.append(current)
    return merged or [code]


# ---------------------------------------------------------------------------
# Fixed Chunking
# ---------------------------------------------------------------------------

def fixed_char_chunk(
    text: str,
    size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Fixed character-count chunks with sliding overlap."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start += size - overlap
    return chunks


def fixed_token_chunk(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
) -> List[str]:
    """
    Token-aware fixed chunking using tiktoken.
    Falls back to character-based approximation (1 token ≈ 4 chars) if unavailable.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            decoded = enc.decode(tokens[start:end])
            if decoded.strip():
                chunks.append(decoded)
            if end >= len(tokens):
                break
            start += max_tokens - overlap_tokens
        return chunks
    except ImportError:
        return fixed_char_chunk(text, size=max_tokens * 4, overlap=overlap_tokens * 4)


# ---------------------------------------------------------------------------
# Recursive Chunking — separator-priority strategy
# ---------------------------------------------------------------------------

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_chunk(
    text: str,
    max_chars: int = 1_000,
    overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursive character-level text splitter.
    Tries each separator in priority order; recurses into sub-splits if a part
    still exceeds max_chars.  Final fallback is hard character splitting.

    Separator priority:  \\n\\n  →  \\n  →  ". "  →  " "  →  ""  (hard split)
    Overlap is added at the seam between consecutive chunks for context continuity.
    """
    if separators is None:
        separators = _DEFAULT_SEPARATORS

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    for sep_idx, sep in enumerate(separators):
        if sep == "":
            # Hard-split fallback
            return fixed_char_chunk(text, max_chars, overlap)

        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks: List[str] = []
        current = ""
        for part in parts:
            joiner = sep if current else ""
            candidate = current + joiner + part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > max_chars:
                    # Recurse into next separator tier
                    sub = recursive_chunk(
                        part,
                        max_chars,
                        overlap,
                        separators[sep_idx + 1:],
                    )
                    if chunks and overlap > 0 and sub:
                        # Prepend overlap from previous chunk
                        tail = chunks[-1][-overlap:]
                        sub[0] = tail + " " + sub[0]
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        if chunks:
            return [c for c in chunks if c.strip()]

    return [text]


# ---------------------------------------------------------------------------
# Document Processor class
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    High-level processor that ties parsing + chunking + metadata + ID generation
    together and pushes results into a VectorEngine instance.

    chunk_strategy: "fixed_char" | "fixed_token" | "recursive" (default)
    """

    def __init__(
        self,
        vector_engine,
        chunk_strategy: str = "recursive",
        chunk_size: int = 1_000,
        chunk_overlap: int = 100,
    ) -> None:
        self.vector_engine = vector_engine
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_all_documents(self, access_level: str = "public") -> Dict[str, Any]:
        """Ingest all Markdown documents from the techcorp-docs directory."""
        total_chunks = 0
        doc_count = 0

        if not DOCS_DIR.exists():
            print(f"  [warn] techcorp-docs directory not found at {DOCS_DIR}")
            return {"documents": 0, "chunks": 0}

        for category_dir in sorted(DOCS_DIR.iterdir()):
            if not category_dir.is_dir():
                continue
            print(f"\n  Processing category: {category_dir.name}")
            for doc_path in sorted(category_dir.glob("*.md")):
                chunks_data = self.process_markdown_file(
                    doc_path,
                    category=category_dir.name,
                    access_level=access_level,
                )
                self._add_to_vector_store(chunks_data)
                total_chunks += len(chunks_data)
                doc_count += 1
                print(f"    {doc_path.name}: {len(chunks_data)} chunks")

        print(f"\n  Total: {doc_count} documents, {total_chunks} chunks ingested")
        return {"documents": doc_count, "chunks": total_chunks}

    def process_markdown_file(
        self,
        path: Path,
        category: str = "general",
        access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """Parse a Markdown file → split by headers → chunk each section."""
        content = path.read_text(encoding="utf-8")
        sections = parse_markdown_by_headers(content)

        raw_pairs: List[Tuple[str, str, int]] = []  # (text, section_name, level)
        for sec in sections:
            for chunk in self._chunk(sec["content"]):
                if chunk.strip():
                    raw_pairs.append((chunk, sec["section"], sec["level"]))

        total = len(raw_pairs)
        chunks_data: List[Dict[str, Any]] = []
        for i, (text, section_name, _) in enumerate(raw_pairs):
            chunk_id = make_chunk_id(str(path), i, text)
            meta = build_metadata(
                source=str(path),
                doc_type="markdown",
                chunk_index=i,
                total_chunks=total,
                section=section_name,
                access_level=access_level,
                extra={
                    "file": path.name,
                    "category": category,
                    "token_count": count_tokens(text),
                },
            )
            chunks_data.append({"id": chunk_id, "text": text, "metadata": meta})

        return chunks_data

    def process_pdf_file(
        self,
        path: Path,
        category: str = "general",
        access_level: str = "public",
        use_ocr: bool = False,
    ) -> List[Dict[str, Any]]:
        """Parse a PDF → extract pages → chunk + annotate with page numbers."""
        pages = parse_pdf_with_ocr(str(path)) if use_ocr else parse_pdf(str(path))

        raw_pairs: List[Tuple[str, int]] = []
        for page_data in pages:
            combined = page_data["text"]
            for tbl in page_data.get("tables", []):
                combined += "\n\n" + tbl
            for chunk in self._chunk(combined):
                if chunk.strip():
                    raw_pairs.append((chunk, page_data["page"]))

        total = len(raw_pairs)
        chunks_data: List[Dict[str, Any]] = []
        for i, (text, page_num) in enumerate(raw_pairs):
            chunk_id = make_chunk_id(str(path), i, text)
            meta = build_metadata(
                source=str(path),
                doc_type="pdf",
                chunk_index=i,
                total_chunks=total,
                page_number=page_num,
                access_level=access_level,
                extra={
                    "file": path.name,
                    "category": category,
                    "token_count": count_tokens(text),
                },
            )
            chunks_data.append({"id": chunk_id, "text": text, "metadata": meta})

        return chunks_data

    def process_html_content(
        self,
        html: str,
        source: str,
        category: str = "web",
        access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """Parse raw HTML → extract prose → chunk."""
        text = parse_html(html, url=source)
        raw_chunks = self._chunk(text)

        total = len(raw_chunks)
        chunks_data: List[Dict[str, Any]] = []
        for i, chunk in enumerate(raw_chunks):
            if not chunk.strip():
                continue
            chunk_id = make_chunk_id(source, i, chunk)
            meta = build_metadata(
                source=source,
                doc_type="html",
                chunk_index=i,
                total_chunks=total,
                access_level=access_level,
                extra={"category": category, "token_count": count_tokens(chunk)},
            )
            chunks_data.append({"id": chunk_id, "text": chunk, "metadata": meta})

        return chunks_data

    def process_code_file(
        self,
        path: Path,
        language: str = "python",
        access_level: str = "public",
    ) -> List[Dict[str, Any]]:
        """Parse a source code file using AST-aware chunking."""
        code = path.read_text(encoding="utf-8")
        raw_chunks = chunk_code(code, language=language, max_chars=self.chunk_size)

        total = len(raw_chunks)
        chunks_data: List[Dict[str, Any]] = []
        for i, chunk in enumerate(raw_chunks):
            if not chunk.strip():
                continue
            chunk_id = make_chunk_id(str(path), i, chunk)
            meta = build_metadata(
                source=str(path),
                doc_type="code",
                chunk_index=i,
                total_chunks=total,
                access_level=access_level,
                extra={
                    "file": path.name,
                    "language": language,
                    "token_count": count_tokens(chunk),
                },
            )
            chunks_data.append({"id": chunk_id, "text": chunk, "metadata": meta})

        return chunks_data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk(self, text: str) -> List[str]:
        if self.chunk_strategy == "fixed_char":
            return fixed_char_chunk(text, size=self.chunk_size, overlap=self.chunk_overlap)
        if self.chunk_strategy == "fixed_token":
            return fixed_token_chunk(text, max_tokens=self.chunk_size // 4, overlap_tokens=self.chunk_overlap // 4)
        # default: recursive
        return recursive_chunk(text, max_chars=self.chunk_size, overlap=self.chunk_overlap)

    def _add_to_vector_store(self, chunks_data: List[Dict[str, Any]]) -> None:
        texts = [c["text"] for c in chunks_data]
        metas = [c["metadata"] for c in chunks_data]
        ids = [c["id"] for c in chunks_data]
        self.vector_engine.add_documents(texts, metas, ids=ids)
