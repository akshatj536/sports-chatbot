from rag.vector_store import VectorStore

# Initialize the VectorStore
store = VectorStore()

# Sample text to add to the vector store
texts = ["Who won the 2022 World Cup?", "What is the capital of France?"]

# Add embeddings to the vector store
store.add_embeddings(texts)

# Save the updated vector store and documents
store.save()

# Search the vector store for a query
query = "Who won the 2022 World Cup?"
results = store.search(query, top_k=3)

# Print top results
print("Top results:")
for result in results:
    print(f"- {result}")
