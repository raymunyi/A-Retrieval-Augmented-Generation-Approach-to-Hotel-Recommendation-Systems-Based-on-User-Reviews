# src/retriever/build_faiss.py
import argparse
import numpy as np
import faiss
import os

def build_index(emb_path, index_path):
    emb = np.load(emb_path)
    d = emb.shape[1]
    # normalize for cosine similarity (use inner product index)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"Built FAISS index at {index_path} with {index.ntotal} vectors (dim={d})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--index", default="data/embeddings/faiss.index")
    args = parser.parse_args()
    build_index(args.emb, args.index)
