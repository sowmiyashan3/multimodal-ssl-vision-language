import torch
import time
import math

EMBED_PATH = "val_embeddings.pt"
data = torch.load(EMBED_PATH)

img_embeds = data["image_embeds"]
txt_embeds = data["text_embeds"]

# Normalize (important for cosine similarity)
img_embeds = img_embeds / img_embeds.norm(dim=1, keepdim=True)
txt_embeds = txt_embeds / txt_embeds.norm(dim=1, keepdim=True)

# Similarity matrix
sim = img_embeds @ txt_embeds.T   # [N, N]
N = sim.size(0)


def recall_at_k(similarity, k):
    correct = 0
    for i in range(N):
        topk = similarity[i].topk(k).indices
        if i in topk:
            correct += 1
    return correct / N

def mean_average_precision(similarity):
    APs = []
    for i in range(N):
        scores = similarity[i]
        ranking = scores.argsort(descending=True)

        # Find rank of correct item
        rank = (ranking == i).nonzero(as_tuple=True)[0].item()

        # Since only 1 relevant item:
        AP = 1.0 / (rank + 1)
        APs.append(AP)

    return sum(APs) / len(APs)

def ndcg_at_k(similarity, k=10):
    ndcgs = []
    for i in range(N):
        scores = similarity[i]
        ranking = scores.argsort(descending=True)

        # Find rank of correct item
        rank = (ranking == i).nonzero(as_tuple=True)[0].item()

        if rank < k:
            dcg = 1.0 / math.log2(rank + 2)
        else:
            dcg = 0.0

        # Ideal DCG = 1 (relevant item at position 0)
        idcg = 1.0

        ndcgs.append(dcg / idcg)

    return sum(ndcgs) / len(ndcgs)

def measure_latency(similarity_matrix):
    start = time.time()

    for i in range(N):
        _ = similarity_matrix[i].topk(10)

    end = time.time()
    avg_latency = (end - start) / N

    return avg_latency


print("Recall@1:", recall_at_k(sim, 1))
print("Recall@5:", recall_at_k(sim, 5))
print("Recall@10:", recall_at_k(sim, 10))

print("mAP:", mean_average_precision(sim))
print("nDCG@10:", ndcg_at_k(sim, 10))

latency = measure_latency(sim)
print(f"Latency per query (flat brute-force): {latency*1000:.4f} ms")
