# src/pipeline/rag_pipeline.py
import os
import sys
# ensure src is on path when running from project root
_this_dir = os.path.dirname(os.path.dirname(__file__))
if _this_dir not in sys.path:
    sys.path.append(_this_dir)

from retriever.retrieve import DPRRetriever
from generator.generate_output import generate_recommendation

def rag_recommend(query, user_type="Tourist", k=5, index_path="data/embeddings/faiss.index", meta_path="data/embeddings/passages_meta.pkl"):
    retriever = DPRRetriever(index_path=index_path, meta_path=meta_path)
    retrieved = retriever.retrieve(query, k=k)
    passages = retrieved["passage_text"].tolist()
    generated = generate_recommendation(user_type, query, passages[:k])
    # also return top hotel names + scores (deduped)
    # small post-processing: group by hotel_name and take highest score
    top = retrieved[["hotel_name", "score"]].copy()
    top = top.groupby("hotel_name", as_index=False).agg({"score":"max"}).sort_values("score", ascending=False).reset_index(drop=True)
    return generated, top

if __name__ == "__main__":
    print("Quick test (ensure embeddings/index exist):")
    text, hotels = rag_recommend("quiet hotel near downtown with good wifi", "Business", k=3)
    print(text)
    print(hotels.head())
