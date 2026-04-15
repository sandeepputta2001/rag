"""
Document Processor - ingests TechCorp documents into the vector store
"""

from pathlib import Path


DOCS_DIR = Path(__file__).parent.parent / "techcorp-docs"


def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += size - overlap
    return chunks


class DocumentProcessor:
    def __init__(self, vector_engine):
        self.vector_engine = vector_engine

    def process_all_documents(self):
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
                content = doc_path.read_text(encoding="utf-8")
                chunks = chunk_text(content)
                metadatas = [
                    {"file": doc_path.name, "category": category_dir.name}
                    for _ in chunks
                ]
                self.vector_engine.add_documents(chunks, metadatas)
                total_chunks += len(chunks)
                doc_count += 1
                print(f"    {doc_path.name}: {len(chunks)} chunks")

        print(f"\n  Total: {doc_count} documents, {total_chunks} chunks ingested")
        return {"documents": doc_count, "chunks": total_chunks}
