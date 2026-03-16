import faiss
import torch
import numpy as np
import os

EMBED_PATH = "../val_embeddings.pt"
INDEX_SAVE_PATH = "faiss.index"

# Load embeddings
data = torch.load(EMBED_PATH)
image_embeds = data["image_embeds"].numpy().astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(image_embeds)

d = image_embeds.shape[1]  # embedding dimension

# --- Build IVF Index (Approximate) ---
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

print("Training FAISS index...")
index.train(image_embeds)

print("Adding vectors to index...")
index.add(image_embeds)

print("Total vectors in index:", index.ntotal)

faiss.write_index(index, INDEX_SAVE_PATH)
print("Index saved to", INDEX_SAVE_PATH)
