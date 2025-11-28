# src/preprocessing/prepare_passages.py
import argparse
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

from clean_text import clean_text

nltk.download("punkt", quiet=True)

def split_into_passages(text, max_words=150):
    """Split long text into passages of up to max_words (approx)."""
    sents = sent_tokenize(text)
    passages = []
    cur = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur_words + w <= max_words:
            cur.append(s)
            cur_words += w
        else:
            if cur:
                passages.append(" ".join(cur))
            cur = [s]
            cur_words = w
    if cur:
        passages.append(" ".join(cur))
    return passages

def prepare(input_csv, output_csv, sample=None, max_passages=None):
    """
    Read dataset, combine Positive_Review and Negative_Review if available,
    create passages and write output.
    - sample: int or None -> number of reviews to sample (random) before passage split
    - max_passages: int or None -> max number of passages to keep overall (useful for laptop)
    """
    print(f"Loading {input_csv} ...")
    # Use low_memory=False to avoid dtype warnings for big CSVs
    df = pd.read_csv(input_csv, low_memory=False)
    print("Columns detected:", list(df.columns))
    # Determine fields
    # Common columns (from your preview): 'Positive_Review', 'Negative_Review', 'Hotel_Name'
    if sample is not None and sample > 0:
        print(f"Sampling {sample} rows from dataset (random)...")
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    rows = []
    total_passage_count = 0
    for idx, r in df.iterrows():
        # make robust column access
        pos = r.get("Positive_Review", "") if "Positive_Review" in r else r.get("Positive Review", "")
        neg = r.get("Negative_Review", "") if "Negative_Review" in r else r.get("Negative Review", "")
        # fallback for some datasets
        if not pos and not neg:
            # try 'Review' variants
            pos = r.get("Review", "") or r.get("Review_Text", "") or r.get("review_text", "")
            neg = ""
        combined = f"{pos} {neg}".strip()
        combined = clean_text(combined)
        if len(combined) < 20:
            continue

        hotel = r.get("Hotel_Name", r.get("Hotel name", r.get("hotel_name", "unknown")))
        hotel = hotel if isinstance(hotel, str) else str(hotel)

        # create passages
        passages = split_into_passages(combined, max_words=150)
        for p_idx, p in enumerate(passages):
            rows.append({
                "orig_index": idx,
                "passage_id": f"{idx}_{p_idx}",
                "hotel_name": hotel,
                "passage_text": p
            })
            total_passage_count += 1
            if max_passages and total_passage_count >= max_passages:
                break
        if max_passages and total_passage_count >= max_passages:
            break

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} passages to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV (tripadvisor.csv)")
    parser.add_argument("--output", required=True, help="Path to output passages.csv")
    parser.add_argument("--sample", type=int, default=None, help="If set, randomly sample N reviews before splitting into passages (good for laptop)")
    parser.add_argument("--max_passages", type=int, default=None, help="If set, stop after producing this many passages (for speed)")
    args = parser.parse_args()
    prepare(args.input, args.output, sample=args.sample, max_passages=args.max_passages)
