import torch

EMBED_PATH = "val_embeddings.pt"

data = torch.load(EMBED_PATH)

img_embeds = data["image_embeds"]
txt_embeds = data["text_embeds"]

# Normalize
img_embeds = img_embeds / img_embeds.norm(dim=1, keepdim=True)
txt_embeds = txt_embeds / txt_embeds.norm(dim=1, keepdim=True)

# Similarity matrix
sim = img_embeds @ txt_embeds.T

def recall_at_k(similarity, k):
    correct = 0
    for i in range(similarity.size(0)):
        topk = similarity[i].topk(k).indices
        if i in topk:
            correct += 1
    return correct / similarity.size(0)

print("Recall@1:", recall_at_k(sim, 1))
print("Recall@5:", recall_at_k(sim, 5))
print("Recall@10:", recall_at_k(sim, 10))
