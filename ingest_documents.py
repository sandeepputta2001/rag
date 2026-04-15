import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("TECHCORP KNOWLEDGE INGESTION SYSTEM")
print("="*50)

# Initialize systems
print("Connecting to AI Brain (from Task 3)...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("techcorp_docs")

print("Loading Semantic Processor (from Task 5)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("All systems online!\n")

# Process documents
print("Beginning knowledge transfer...")
doc_count = 0
total_chunks = 0

docs_root = Path(__file__).parent / 'techcorp-docs'
for category in docs_root.iterdir():
    if category.is_dir():
        print(f"\nProcessing {category.name}:")

        for doc in category.glob('*.md'):
            print(f"  {doc.name}", end="")

            with open(doc, 'r') as f:
                content = f.read()

            # Apply chunking strategy from Task 4!
            chunks = [content[i:i+500] for i in range(0, len(content), 400)]

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc.stem}_{i}"
                # Apply embedding from Task 5!
                embedding = model.encode(chunk).tolist()

                # Store in database from Task 3!
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas={"file": doc.name, "category": category.name}
                )
                total_chunks += 1

            doc_count += 1
            print(f" ({len(chunks)} chunks)")

print("\n" + "="*50)
print(f"INGESTION COMPLETE!")
print(f"Statistics:")
print(f"   • Documents processed: {doc_count}")
print(f"   • Knowledge chunks: {total_chunks}")
print(f"   • AI IQ increased: +{doc_count*10} points")
print(f"\nValue delivered: $500K in searchable knowledge!")

# Save results
with open(Path(__file__).parent / 'ingest-complete.txt', 'w') as f:
    f.write(f"DOCS:{doc_count},CHUNKS:{collection.count()}")