# src/retriever/dpr_embed.py
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
import os
from tqdm import tqdm

def embed_passages(passages_csv, out_emb, out_meta, batch_size=16):
    df = pd.read_csv(passages_csv)
    texts = df["passage_text"].astype(str).tolist()
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding passages"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).pooler_output.cpu().numpy()
        all_emb.append(out)
    emb = np.vstack(all_emb)
    os.makedirs(os.path.dirname(out_emb), exist_ok=True)
    np.save(out_emb, emb)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    df.to_pickle(out_meta)
    print(f"Saved embeddings: {out_emb} ({emb.shape}) and metadata: {out_meta}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="passages csv")
    parser.add_argument("--out_emb", default="data/embeddings/passages.npy")
    parser.add_argument("--out_meta", default="data/embeddings/passages_meta.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    embed_passages(args.input, args.out_emb, args.out_meta, args.batch_size)
