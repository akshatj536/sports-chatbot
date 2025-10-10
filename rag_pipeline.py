# rag_pipeline.py

from transformers import pipeline
from rag.vector_store import VectorStore
from rank_bm25 import BM25Okapi  # Import BM25
import numpy as np
import json


# Load the QA model (Flan-T5 for generation)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Initialize Vector Store
vector_store = VectorStore(dimension=384)  # Already handles loading index + documents

# Load static data for BM25 search\\
with open("mock_data/mock_football_data.json", "r") as f:
    static_documents = json.load(f)

# Prepare BM25 for static text-based retrieval
static_texts = [doc['text'] for doc in static_documents]
bm25 = BM25Okapi([text.split() for text in static_texts])


def rag_pipeline(query: str, top_k: int = 20 ) -> str:
    """
    Retrieves top-k relevant documents using hybrid search (BM25 + vector search) and generates an answer using those.
    
    Args:
        query (str): User question.
        top_k (int): Number of relevant documents to fetch.
    
    Returns:
        str: Generated answer.
    """
    # Step 1: Retrieve top-k relevant documents using BM25 (static search)
    bm25_results = bm25.get_top_n(query.split(), static_texts, n=top_k)

    # Step 2: Retrieve top-k relevant documents using vector search (dynamic + static)
    vector_search_results = vector_store.search(query, top_k=top_k)

    # Extract only the text from both results
    bm25_texts = [doc for doc in bm25_results]  # Assuming bm25_results is a list of strings
    vector_search_texts = [result for result in vector_search_results]  # Assuming vector_search_results is a list of dicts

    # Combine BM25 and vector search results, remove duplicates (using list comprehension)
    # combined_results = list(set(bm25_texts + vector_search_texts))[:top_k]
    combined_results=vector_search_results
    print(combined_results)
    # Format the combined results for context
    # context = " ".join(combined_results)

    # # Step 3: Formulate prompt for generation
    # prompt = f"Context: {context} \nQuestion: {query} \nAnswer:"

    # # Step 4: Use model to generate the answer
    # generated_response = qa_pipeline(prompt)[0]["generated_text"]

    # return generated_response

if __name__ == "__main__":
    query = "about Bayern Munich"
    response = rag_pipeline(query)
    print("\nGenerated Answer:\n", response)
