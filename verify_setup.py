"""Verification script to test all components work correctly."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        import fastapi
        import uvicorn
        import numpy as np
        import sentence_transformers
        import faiss
        import sklearn
        print("  ✓ All dependencies installed correctly")
        return True
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        return False


def test_embedding_db():
    """Test embedding database."""
    print("\nTesting embedding database...")
    try:
        from embedding_db import EmbeddingDatabase
        db = EmbeddingDatabase()
        
        # Just check model loads
        print(f"  ✓ Embedding model loaded: {db.model}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_fuzzy_clustering():
    """Test fuzzy clustering."""
    print("\nTesting fuzzy clustering...")
    try:
        from fuzzy_clustering import FuzzyClustering
        from config import N_CLUSTERS
        
        clustering = FuzzyClustering(n_clusters=N_CLUSTERS)
        print(f"  ✓ Clustering initialized with {N_CLUSTERS} clusters")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_semantic_cache():
    """Test semantic cache."""
    print("\nTesting semantic cache...")
    try:
        from semantic_cache import SemanticCache
        import numpy as np
        
        cache = SemanticCache()
        
        # Add a dummy entry
        dummy_emb = np.random.randn(384).astype(np.float32)
        dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)
        
        cache.add("test query", dummy_emb, "test result", 0, np.ones(12) / 12)
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        
        print(f"  ✓ Cache works: {stats['total_entries']} entry stored")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_api_structure():
    """Test API structure."""
    print("\nTesting API structure...")
    try:
        from api import app
        
        # Check routes exist
        routes = [route.path for route in app.routes]
        required_routes = ['/query', '/cache/stats', '/cache', '/health']
        
        for route in required_routes:
            if route not in routes:
                print(f"  ✗ Missing route: {route}")
                return False
        
        print(f"  ✓ All required API routes present")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        from config import (
            EMBEDDING_MODEL,
            EMBEDDING_DIMENSION,
            N_CLUSTERS,
            CACHE_SIMILARITY_THRESHOLD,
            API_PORT,
        )
        
        assert EMBEDDING_DIMENSION == 384
        assert N_CLUSTERS == 12
        assert 0.5 < CACHE_SIMILARITY_THRESHOLD < 1.0
        assert API_PORT == 8000
        
        print(f"  ✓ Configuration valid")
        print(f"    - Embedding model: {EMBEDDING_MODEL}")
        print(f"    - Clusters: {N_CLUSTERS}")
        print(f"    - Cache threshold: {CACHE_SIMILARITY_THRESHOLD}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("SEMANTIC SEARCH SYSTEM - COMPONENT VERIFICATION")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_embedding_db,
        test_fuzzy_clustering,
        test_semantic_cache,
        test_api_structure,
        test_config,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Unexpected error in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    if all(results):
        print("✅ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: python src/download_dataset.py")
        print("   (This loads the 20 Newsgroups dataset and builds indices)")
        print("\n2. Then start the API:")
        print("   python -m uvicorn src.api:app --reload")
        print("\n3. Open http://localhost:8000/docs for interactive testing")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
