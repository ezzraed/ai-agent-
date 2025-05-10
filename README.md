# ai-agent-
# AI Agent for Natural Language Querying of Financial Transactions

## 🧠 Summary
This project is a simple Retrieval-Augmented Generation (RAG) system that allows a user to ask natural language questions about their financial transaction history. The system uses sentence embeddings to semantically search relevant records from a CSV file, and a transformer model is used to generate a text-based answer based on the context of the retrieved documents.

## 🔧 Dependencies
You'll need to install the following Python packages for the system to work:

```bash
pip install pandas transformers sentence-transformers faiss-cpu

📂 Dataset

This project requires a CSV file named jordan_transactions.csv. The file should have at least the following columns:

transaction_date: The date of the transaction.
transaction_amount: The amount spent in the transaction.
transaction_type: The category or type of the transaction.
(Modify the column names if your dataset uses different headers.)






🧱 Code Breakdown

1. Loading and Formatting the Dataset
import pandas as pd

# Load your transaction CSV file
df = pd.read_csv("jordan_transactions.csv")

# Format each transaction into a natural language sentence
documents = df.apply(
    lambda row: f"On {row['transaction_date']}, {row['transaction_amount']} was spent on {row['transaction_type']}",
    axis=1
).tolist()









2. Generate Sentence Embeddings
from sentence_transformers import SentenceTransformer

# Load a sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate dense vector representations for each sentence
embeddings = embedder.encode(documents, convert_to_tensor=False)







3. Building a FAISS Search Index
import faiss
import numpy as np

# Set up a FAISS index for fast similarity search
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save the mapping between document indices and original text
id_to_doc = {i: doc for i, doc in enumerate(documents)}








4. Load a Generative Language Model
from transformers import pipeline

# Load a generative transformer model
rag_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")











5. Query Function
def query_rag_system(query, top_k=5):
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)

    retrieved_docs = [id_to_doc[i] for i in I[0]]
    context = "\n".join(retrieved_docs)

    input_text = f"Context:\n{context}\n\nQuestion: {query}"
    response = rag_pipeline(input_text, max_new_tokens=200)[0]['generated_text']
    return response











✅ Example Queries

Here are some example queries you can ask the system:

print(query_rag_system("What were the largest transactions at C Mall?"))
print(query_rag_system("How much was spent on groceries in April?"))
print(query_rag_system("Show me all failed transactions at Z Mall last week?"))








"Added detailed documentation to README"
