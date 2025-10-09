import faiss
import json
import numpy as np
from typing import List, Union
import os
from rag.embedder import get_embedding

class VectorStore:
    def __init__(self, 
                 dimension: int = 384, 
                 index_path: str = "faiss.index", 
                 static_doc_path: str = "mock_football_data.json", 
                 dynamic_doc_path: str = "dynamic_data.json",
                 documents_path: str = "documents.json"):

        self.dimension = dimension
        self.index_path = index_path
        self.static_doc_path = static_doc_path
        self.dynamic_doc_path = dynamic_doc_path
        self.documents_path = documents_path
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

        # Load FAISS index if it exists
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded index from {self.index_path}")
            except Exception as e:
                print(f"Failed to load index: {e}")

        # Load documents from both sources
        self.documents = self._load_documents()
        print(f"Loaded {len(self.documents)} documents (static + dynamic)")

    def _load_documents(self) -> List[str]:
        documents = []
        for path in [self.static_doc_path, self.dynamic_doc_path]:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        docs = json.load(f)
                        if isinstance(docs, list):
                            documents.extend(docs)
                except Exception as e:
                    print(f"Failed to load documents from {path}: {e}")
        return documents

    def add_embeddings(self, texts: List[str]) -> None:
        embeddings = np.array([get_embedding(text) for text in texts], dtype=np.float32)
        self.index.add(embeddings)
        self.documents.extend(texts)
        print(f"Added {len(texts)} embeddings to the index.")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = np.array([get_embedding(query)], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        print(self.index.search(query_embedding, top_k))
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        # save static and dynamic documents to documents.json
        with open(self.static_doc_path, "w") as f:
            json.dump(self.documents, f)
        print(f"Index saved to {self.index_path}")
        print(f"Documents saved to {self.static_doc_path}")