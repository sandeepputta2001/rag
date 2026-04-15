from sentence_transformers import SentenceTransformer
import numpy as np

print(" Loading Google's AI Brain (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Brain loaded! 90M parameters ready!\n")

# TechCorp test sentences
sentences = [
    "Dogs are allowed in the office on Fridays",
    "Pets can come to work on Furry Fridays",
    "Remote work policy allows 3 days from home"
]

print(" Converting text to vectors...")
embeddings = model.encode(sentences)
print(f" Created {len(embeddings)} vectors of {len(embeddings[0])} dimensions each!\n")

# Calculate semantic similarities
sim_1_2 = np.dot(embeddings[0], embeddings[1])
sim_1_3 = np.dot(embeddings[0], embeddings[2])

print(" Semantic Similarity Analysis:")
print("="*50)
print(f"'Dogs allowed' ←→ 'Pets permitted'")
print(f"Similarity: {sim_1_2:.3f} (Very Related! )\n")

print(f"'Dogs allowed' ←→ 'Remote work'")
print(f"Similarity: {sim_1_3:.3f} (Not Related )\n")

# Visualization
print(" Similarity Scale:")
print("0.0  1.0")
print(f"     Remote {'' * int(sim_1_3*20)}")
print(f"     Pets   {'' * int(sim_1_2*20)}")

# Save results
with open('/root/embedding-test.txt', 'w') as f:
    f.write(f"SIM_PET:{sim_1_2:.3f},SIM_REMOTE:{sim_1_3:.3f}")

print("\n You've unlocked semantic understanding!")