import os

print(" DOCUMENT CHUNKING ENGINE")
print("="*40)

def chunk_text(text, size=500, overlap=100):
    """Smart chunking with overlap for context preservation"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += size - overlap

    return chunks

# Process sample document
sample_doc = """TechCorp Pet Policy: 
Employees may bring pets to the office on Fridays. 
Dogs must be well-behaved and vaccinated. 
The CEO's golden retriever is the office mascot.

Remote Work Policy:
Employees can work remotely up to 3 days per week.
Core hours are 10 AM - 3 PM in your local timezone.
All meetings should be recorded for async collaboration.

Benefits Overview:
Comprehensive health insurance including dental and vision.
401k matching up to 6% of salary.
Unlimited PTO after first year.
Annual learning budget of $2,000."""

print(f" Original document: {len(sample_doc)} characters")
print("-"*40)

chunks = chunk_text(sample_doc, size=500, overlap=100)

print(f" Created {len(chunks)} chunks from {len(sample_doc)} chars")
print(" Ready for vectorization!")