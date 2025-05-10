
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Title
st.title("Bots in Black - Financial Query Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload your transactions CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show data preview
    st.subheader("Data Preview")
    st.write(df.head())

    # Format text for embedding
    def safe_format(row):
        return f"Date: {row.get('Transaction Date', '')}, Amount: {row.get('Amount', '')}, Category: {row.get('Category', '')}, Description: {row.get('Details', '')}"

    documents = df.apply(safe_format, axis=1).tolist()

    # Load embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed documents
    embeddings = embedder.encode(documents, convert_to_tensor=False)

    # FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    id_to_doc = {i: doc for i, doc in enumerate(documents)}

    # Load text generation model
    rag_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    # User input
    st.subheader("Ask a Question About Your Transactions")
    user_query = st.text_input("Type your question here:")

    if user_query:
        # Embed query
        query_embedding = embedder.encode([user_query])[0]
        D, I = index.search(np.array([query_embedding]), 5)
        retrieved_docs = [id_to_doc[i] for i in I[0]]

        st.subheader("Retrieved Context")
        for doc in retrieved_docs:
            st.markdown(f"- {doc}")

        # Generate answer
        context = "\n".join(retrieved_docs)
        input_text = f"Context:\n{context}\n\nQuestion: {user_query}"
        response = rag_pipeline(input_text, max_length=100, do_sample=False)[0]["generated_text"]

        st.subheader("Generated Answer")
        st.success(response)
