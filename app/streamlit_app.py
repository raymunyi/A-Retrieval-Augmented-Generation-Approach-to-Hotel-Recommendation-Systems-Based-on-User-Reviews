# app/streamlit_app.py
import streamlit as st
import pandas as pd
import sys
import os
# Ensure src is findable
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.append(os.path.join(proj_root, "src"))

from pipeline.rag_pipeline import rag_recommend

st.set_page_config(page_title="Hotel RAG Recommender", layout="wide")
st.title("Hotel RAG Recommender")

st.markdown("Enter your preferences and select traveler type. The system will retrieve review passages and generate explainable recommendations.")

user_type = st.selectbox("Traveler Type", ["Tourist", "Business"])
query = st.text_input("Enter hotel preferences (in plain English)", "Quiet hotel near downtown with good Wi-Fi")

k = st.slider("Number of retrieved passages to use", min_value=1, max_value=10, value=5)

if st.button("Recommend"):
    with st.spinner("Running retrieval and generation (may take a few seconds)..."):
        try:
            text, hotels = rag_recommend(query, user_type, k=k)
            st.subheader("Generated Explanation / Recommendations")
            st.write(text)

            st.subheader("Top Retrieved Hotels (deduped)")
            st.dataframe(hotels.head(10))
        except Exception as e:
            st.error(f"Error when running pipeline: {e}")
            st.stop()
