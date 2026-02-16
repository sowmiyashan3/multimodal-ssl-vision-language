import faiss
import torch
import numpy as np
import time

INDEX_PATH = "faiss.index"
EMBED_PATH = "../val_embeddings.pt"

# Load index
index = faiss.read_index(INDEX_PATH)

print("Index type:", type(index))
print("Total vectors:", index.ntotal)

# Load embeddings
data = torch.load(EMBED_PATH)
text_embeds = data["text_embeds"].numpy().astype("float32")

faiss.normalize_L2(text_embeds)

# Set search parameters
index.nprobe = 10  # number of clusters to search

k = 5

# --- Measure latency ---
num_queries = 100
start = time.time()

for i in range(num_queries):
    query = text_embeds[i:i+1]
    D, I = index.search(query, k)

end = time.time()

avg_time = (end - start) / num_queries

print("Top-k indices for first query:", I)
print(f"Average search time per query: {avg_time:.6f} seconds")


correct = 0
total = text_embeds.shape[0]

for i in range(total):
    query = text_embeds[i:i+1]
    _, indices = index.search(query, 10)
    if i in indices[0]:
        correct += 1

recall10 = correct / total
print("FAISS Recall@10:", recall10)