# Multimodal Self-Supervised Vision–Language Embeddings

This project implements a **CLIP-style multimodal learning pipeline** that learns aligned embeddings between images and text captions using **contrastive self-supervised learning**.

The system is trained on the **Conceptual Captions dataset** and produces a shared embedding space where semantically related images and captions are close together. The learned embeddings are then used to build a **FAISS-based vector search system** for fast semantic image retrieval.

---

## Architecture

The system follows a **dual-encoder architecture**:

Image → Vision Transformer (ViT) → Image Embedding
Caption → BERT Encoder → Text Embedding

Image Embedding ↔ Text Embedding
↓
Contrastive Learning (InfoNCE Loss)
↓
Shared Multimodal Embedding Space
↓
FAISS Vector Index
↓
Semantic Image Retrieval


---

## Key Features

- **Multimodal representation learning** using contrastive self-supervised training  
- **Vision Transformer (ViT)** as the image encoder  
- **BERT** as the text encoder  
- Shared embedding space for image–text alignment  
- **FAISS approximate nearest neighbor search** for scalable retrieval  
- Retrieval evaluation using **Recall@K, mAP, and nDCG**

---

## Project Pipeline

1. **Data Ingestion**
   - Image–caption pairs downloaded from AWS S3
   - Dataset stored locally with CSV metadata

2. **Model Training**
   - Images encoded using **Vision Transformer**
   - Captions encoded using **BERT**
   - Embeddings aligned using **contrastive InfoNCE loss**

3. **Embedding Generation**
   - Trained model generates embeddings for validation dataset

4. **Evaluation**
   - Retrieval quality measured using Recall@K, mAP, and nDCG

5. **Vector Search**
   - Embeddings indexed using **FAISS IVF index**
   - Enables fast semantic image search

---

## Tech Stack

**Languages & Frameworks**

- Python
- PyTorch
- HuggingFace Transformers

**Libraries**

- Torchvision
- FAISS
- NumPy
- Pandas

**Infrastructure**

- AWS S3
- GPU training

---


---

## Applications

This system can be used for:

- semantic image search
- multimodal recommendation systems
- visual search engines
- multimodal retrieval-augmented generation (RAG)
- zero-shot image classification

---

## Author

**Sowmiya Shanmugamurthy**  
Master’s in Computer Science  
University of Texas at Dallas
