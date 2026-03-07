"""Download and prepare the 20 Newsgroups dataset for the semantic search system."""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from .embedding_db import init_embedding_db
from .fuzzy_clustering import init_fuzzy_clustering
from .config import VECTOR_DB_PATH, CLUSTERING_MODEL_PATH


def main():
    """
    Main setup function.
    
    This does the heavy lifting once:
    1. Downloads 20 Newsgroups if needed
    2. Preprocesses documents
    3. Creates embeddings for all documents
    4. Builds FAISS index for fast similarity search
    5. Trains fuzzy clustering model
    
    Takes 5-10 minutes depending on machine (mostly embedding step).
    Results are cached to disk for fast startup on subsequent runs.
    """
    print("=" * 70)
    print("SEMANTIC SEARCH SYSTEM - INITIALIZATION")
    print("=" * 70)
    print()
    
    # Check if already built
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(CLUSTERING_MODEL_PATH):
        print("✅ System already initialized!")
        print(f"   - Vector database: {VECTOR_DB_PATH}")
        print(f"   - Clustering model: {CLUSTERING_MODEL_PATH}")
        print()
        print("To rebuild from scratch, delete these files:")
        print(f"   - rm data/vector_db.faiss")
        print(f"   - rm data/clustering_model.pkl")
        return
    
    print("🔍 Step 1: Loading and preprocessing dataset...")
    print("-" * 70)
    db = init_embedding_db()
    print(f"✅ Loaded {len(db)} documents")
    print()
    
    print("🎯 Step 2: Training fuzzy clustering model...")
    print("-" * 70)
    clustering = init_fuzzy_clustering(db.embeddings)
    print()
    
    # Print cluster analysis
    print("📊 Cluster Analysis:")
    print("-" * 70)
    for cluster_id in range(clustering.n_clusters):
        info = clustering.get_cluster_info(cluster_id, db.embeddings)
        print(f"Cluster {cluster_id:2d}: {info['num_documents']:4d} documents, "
              f"avg membership={info['avg_membership_strength']:.2f}")
    print()
    
    # Find boundary cases
    print("🔍 Analyzing Boundary Cases (most ambiguous documents):")
    print("-" * 70)
    boundary_docs = clustering.analyze_boundaries(db.texts, db.embeddings)
    
    for i, doc_info in enumerate(boundary_docs[:5], 1):
        doc_idx = doc_info['doc_idx']
        text = db.texts[doc_idx][:150]
        print(f"{i}. Doc {doc_idx}:")
        print(f"   Clusters {doc_info['primary_cluster']}/{doc_info['secondary_cluster']} "
              f"[{doc_info['primary_prob']:.2f}/{doc_info['secondary_prob']:.2f}]")
        print(f"   '{text}...'")
        print()
    
    print("=" * 70)
    print("✅ INITIALIZATION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Start the API:")
    print("     python -m uvicorn src.api:app --reload")
    print()
    print("  2. Open http://localhost:8000/docs in your browser")
    print()
    print("  3. Try a query:")
    print("     POST /query")
    print("     {'query': 'What are good graphics cards for gaming?'}")
    print()


if __name__ == "__main__":
    main()
