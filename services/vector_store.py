import chromadb
from chromadb.config import Settings
from typing import List
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.chroma_db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection('documents')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_documents(self, documents: List[Document]):
        texts = [doc.page_content for doc in documents]
        embeddings = self.encoder.encode(texts).tolist()

        ids = [f'doc_{i}' for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 5) -> List[Document]:
        query_embedding = self.encoder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        documents = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            documents.append(doc)

        return documents