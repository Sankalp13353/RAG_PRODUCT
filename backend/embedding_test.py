from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How does authentication work?",
    "Where is the user login system implemented?",
    "How is the AQI prediction calculated?",
]

embeddings = model.encode(sentences)

print(f"Embedding dimensions: {len(embeddings[0])}\n")

sim_1_2 = float(util.cos_sim(embeddings[0], embeddings[1]))
sim_1_3 = float(util.cos_sim(embeddings[0], embeddings[2]))

print(f"Cosine similarity (Sentence 1 & Sentence 2): {sim_1_2:.4f}")
print(f"Cosine similarity (Sentence 1 & Sentence 3): {sim_1_3:.4f}")
