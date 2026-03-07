"""Vector database using FAISS for efficient similarity search."""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pickle
import os
from .config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, VECTOR_DB_PATH, EMBEDDINGS_CACHE_PATH
from .dataset import load_and_preprocess_dataset


class EmbeddingDatabase:
    """
    FAISS-based vector database for semantic search.
    
    Design rationale:
    - FAISS: Sub-linear search, GPU-friendly, production-tested
    - Cosine similarity: More suitable than L2 for text embeddings
    - Pre-computed and cached: Embedding is expensive, only do once
    """
    
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.embeddings = None
        self.texts = None
        self.raw_texts = None
        self.category_ids = None
        self.document_ids = None
    
    def build(self, texts: List[str], raw_texts: List[str], category_ids: List[int]):
        """
        Build index from raw texts.
        
        Args:
            texts: Preprocessed texts for embedding
            raw_texts: Original texts for reference
            category_ids: Original newsgroup categories
        """
        print(f"Embedding {len(texts)} documents using {EMBEDDING_MODEL}...")
        
        self.texts = texts
        self.raw_texts = raw_texts
        self.category_ids = np.array(category_ids)
        
        # Embed all texts
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.asarray(self.embeddings).astype('float32')
        
        # Normalize for cosine similarity
        from sklearn.preprocessing import normalize
        self.embeddings = normalize(self.embeddings, norm='l2')
        
        # Create FAISS index
        # IndexFlatIP: Inner product (cosine after normalization)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.index.add(self.embeddings)
        
        self.document_ids = np.arange(len(texts))
        
        print(f"Built index with {self.index.ntotal} documents")
        
        # Save to disk
        self._save()
    
    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[float], List[int]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Tuple of (documents, similarities, doc_indices)
        """
        query_embedding = self.model.encode(query)
        query_embedding = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        from sklearn.preprocessing import normalize
        query_embedding = normalize(query_embedding, norm='l2')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # distances are cosine similarities after normalization
        similarities = distances[0].tolist()
        doc_indices = indices[0].tolist()
        documents = [self.raw_texts[idx] for idx in doc_indices]
        
        return documents, similarities, doc_indices
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        embedding = self.model.encode(text)
        return np.asarray(embedding, dtype='float32')
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings, dtype='float32')
    
    def _save(self):
        """Save index and embeddings to disk."""
        os.makedirs(os.path.dirname(VECTOR_DB_PATH) or '.', exist_ok=True)
        
        faiss.write_index(self.index, VECTOR_DB_PATH)
        np.save(EMBEDDINGS_CACHE_PATH, self.embeddings)
        
        # Save metadata
        metadata = {
            'texts': self.texts,
            'raw_texts': self.raw_texts,
            'category_ids': self.category_ids,
            'document_ids': self.document_ids,
        }
        with open('data/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved FAISS index to {VECTOR_DB_PATH}")
    
    def load(self):
        """Load index from disk."""
        if not os.path.exists(VECTOR_DB_PATH):
            return False
        
        self.index = faiss.read_index(VECTOR_DB_PATH)
        self.embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        
        with open('data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.texts = metadata['texts']
        self.raw_texts = metadata['raw_texts']
        self.category_ids = metadata['category_ids']
        self.document_ids = metadata['document_ids']
        
        print(f"Loaded FAISS index with {self.index.ntotal} documents")
        return True
    
    def get_document_text(self, doc_id: int) -> str:
        """Get original document text by ID."""
        return self.raw_texts[doc_id]
    
    def get_document_embedding(self, doc_id: int) -> np.ndarray:
        """Get embedding for a document by ID."""
        return self.embeddings[doc_id]
    
    def __len__(self):
        return self.index.ntotal if self.index else 0


def init_embedding_db() -> EmbeddingDatabase:
    """Initialize embedding database, building if necessary."""
    db = EmbeddingDatabase()
    
    # Try to load from cache
    if db.load():
        return db
    
    # Build from scratch
    texts, raw_texts, category_ids = load_and_preprocess_dataset()
    db.build(texts, raw_texts, category_ids)
    
    return db


if __name__ == "__main__":
    db = init_embedding_db()
    
    # Test search
    query = "What is the difference between guns and laws about guns?"
    docs, sims, indices = db.search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Top 3 results:")
    for i, (doc, sim, idx) in enumerate(zip(docs, sims, indices)):
        print(f"\n{i+1}. Similarity: {sim:.3f} (Doc {idx})")
        print(f"   {doc[:200]}...")
