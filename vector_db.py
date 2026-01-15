# import os
# import pickle
# import faiss
# import numpy as np
# from typing import List, Dict, Optional
# from dotenv import load_dotenv
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Load environment variables
# load_dotenv()


# class VectorStore:
#     """FAISS-based vector store with Azure OpenAI embeddings"""
    
#     def __init__(self):
#         endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#         api_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
#         deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
#         api_version = "2024-12-01-preview"
#         self.dimension = 1536
#         class LocalEmbeddings:
#             def __init__(self, dim: int):
#                 self.dim = dim
#             def _vec(self, text: str) -> np.ndarray:
#                 seed = abs(hash(text)) % (2**32)
#                 rng = np.random.default_rng(seed)
#                 v = rng.normal(0, 1, self.dim).astype('float32')
#                 v /= max(np.linalg.norm(v), 1e-6)
#                 return v
#             def embed_documents(self, texts: List[str]) -> List[List[float]]:
#                 return [self._vec(t).tolist() for t in texts]
#             def embed_query(self, text: str) -> List[float]:
#                 return self._vec(text).tolist()
#         try:
#             if endpoint and api_key and deployment_name:
#                 self.embeddings = AzureOpenAIEmbeddings(
#                     deployment=deployment_name,
#                     azure_endpoint=endpoint,
#                     openai_api_key=api_key,
#                     openai_api_version=api_version,
#                 )
#             else:
#                 self.embeddings = LocalEmbeddings(self.dimension)
#         except Exception:
#             self.embeddings = LocalEmbeddings(self.dimension)
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,  
#             chunk_overlap=200,
#             length_function=len,
#         )
#         self.index = None
#         self.metadata_store: Dict[int, Dict] = {}
#         self.next_id = 0
        
#     def _init_index(self):
#         """Initialize FAISS index if not exists"""
#         if self.index is None:
#             self.index = faiss.IndexFlatL2(self.dimension)
    
#     def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
#         """
#         Add documents to vector store
#         Returns: List of document IDs
#         """
        
#         chunks = []
#         chunk_metadatas = []
        
#         for i, text in enumerate(texts):
#             splits = self.text_splitter.split_text(text)
#             chunks.extend(splits)
#             metadata = metadatas[i] if metadatas else {}
#             chunk_metadatas.extend([{**metadata, "chunk_index": j} for j in range(len(splits))])
        
#         if not chunks:
#             return []
        
#         try:
#             embeddings_list = self.embeddings.embed_documents(chunks)
#         except Exception as e:
#             raise Exception(f"Error generating embeddings: {str(e)}")
        
#         self._init_index()
        
#         # Add to FAISS index
#         embeddings_array = np.array(embeddings_list).astype('float32')
#         self.index.add(embeddings_array)
        
#         # Store metadata
#         doc_ids = []
#         for i, (chunk, metadata) in enumerate(zip(chunks, chunk_metadatas)):
#             doc_id = str(self.next_id)
#             self.metadata_store[self.next_id] = {
#                 "text": chunk,
#                 "metadata": metadata,
#             }
#             doc_ids.append(doc_id)
#             self.next_id += 1
        
#         return doc_ids
    
#     def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
#         """
#         Search for similar documents
#         Returns: List of dictionaries with 'text' and 'metadata' keys
#         """
#         if self.index is None or self.index.ntotal == 0:
#             return []
        
#         # Generate query embedding
#         try:
#             query_embedding = self.embeddings.embed_query(query)
#         except Exception as e:
#             raise Exception(f"Error generating query embedding: {str(e)}")
        
#         # Search in FAISS
#         query_vector = np.array([query_embedding]).astype('float32')
#         distances, indices = self.index.search(query_vector, k)
        
#         # Retrieve results
#         results = []
#         for idx, distance in zip(indices[0], distances[0]):
#             if idx < 0:  # FAISS returns -1 for empty results
#                 continue
#             if idx in self.metadata_store:
#                 result = {
#                     "text": self.metadata_store[idx]["text"],
#                     "metadata": self.metadata_store[idx]["metadata"],
#                     "distance": float(distance)
#                 }
#                 results.append(result)
        
#         return results
    
#     def clear(self):
#         """Clear the vector store"""
#         self.index = None
#         self.metadata_store = {}
#         self.next_id = 0
    
#     def save(self, filepath: str):
#         """Save vector store to disk"""
#         if self.index is None:
#             return
        
#         os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
#         # Save FAISS index
#         faiss.write_index(self.index, filepath + ".index")
        
#         # Save metadata
#         with open(filepath + ".meta", "wb") as f:
#             pickle.dump({
#                 "metadata_store": self.metadata_store,
#                 "next_id": self.next_id,
#             }, f)
    
#     def load(self, filepath: str):
#         """Load vector store from disk"""
#         if not os.path.exists(filepath + ".index"):
#             return False
        
#         # Load FAISS index
#         self.index = faiss.read_index(filepath + ".index")
#         self.dimension = self.index.d
        
#         # Load metadata
#         with open(filepath + ".meta", "rb") as f:
#             data = pickle.load(f)
#             self.metadata_store = data["metadata_store"]
#             self.next_id = data["next_id"]
        
#         return True


# # Global vector store instance (per-user stores will be managed in main.py)
# def create_vector_store() -> VectorStore:
#     """Factory function to create a new vector store instance"""
#     return VectorStore()

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import requests

load_dotenv()

class VectorDB:
    def __init__(self, username: str):
        self.username = username  # This will be our "Namespace"
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Use OpenAI Embeddings (dimension 1536)
        # Assumes OPENAI_API_KEY is in env
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )
        
        # Connect to Pinecone
        # The vector_store object handles the connection logic automatically
        # We just need to initialize it with the index name and namespace
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.username  # CRITICAL: Separates users
        )

    def add_documents(self, texts, metadatas=None):
        """
        Splits text and upserts vectors to Pinecone
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        # Create Document objects
        docs = text_splitter.create_documents(texts, metadatas=metadatas)
        
        # Upload to Pinecone
        return self.vector_store.add_documents(docs)

    def similarity_search(self, query, k=4):
        """
        Search Pinecone for similar vectors
        """
        # Search specifically in this user's namespace
        return self.vector_store.similarity_search_with_score(query, k=k)
        
    def clear(self):
        """
        Delete all vectors for this user
        """
        # Pinecone allows deleting by namespace, instantly wiping user data
        self.vector_store.delete(delete_all=True, namespace=self.username)

# Factory function to keep your main.py clean
def create_vector_store(username):
    return VectorDB(username)
