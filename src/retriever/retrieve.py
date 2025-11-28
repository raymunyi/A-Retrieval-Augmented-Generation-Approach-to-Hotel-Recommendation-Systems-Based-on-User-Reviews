# src/retriever/retrieve.py
import faiss
import pandas as pd
import numpy as np
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

class DPRRetriever:
    def __init__(self, index_path="data/embeddings/faiss.index", meta_path="data/embeddings/passages_meta.pkl"):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(index_path)
        self.meta = pd.read_pickle(meta_path)
        # ensure index-normalization assumptions: index uses normalized vectors
        self.tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def retrieve(self, query, k=5):
        enc = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64).to(self.device)
        with torch.no_grad():
            qvec = self.model(**enc).pooler_output.cpu().numpy()
        faiss.normalize_L2(qvec)
        scores, idxs = self.index.search(qvec, k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()
        results = self.meta.iloc[idxs].copy().reset_index(drop=True)
        results["score"] = scores
        return results
