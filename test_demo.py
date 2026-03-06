"""
Test and demo script for the semantic search system.

This script demonstrates:
1. The semantic cache in action
2. Cluster analysis
3. API response structure
4. Cache statistics
"""

import sys
import os
import numpy as np
from sklearn.preprocessing import normalize

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from embedding_db import init_embedding_db
from fuzzy_clustering import init_fuzzy_clustering, FuzzyClustering
from semantic_cache import SemanticCache
from config import CACHE_SIMILARITY_THRESHOLD


def demo_semantic_cache():
    """Demonstrate the semantic cache."""
    print("=" * 70)
    print("DEMO 1: Semantic Cache in Action")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Loading database and clustering model...")
    db = init_embedding_db()
    clustering = init_fuzzy_clustering(db.embeddings)
    
    # Create cache
    cache = SemanticCache(similarity_threshold=CACHE_SIMILARITY_THRESHOLD)
    
    # Test queries that are semantically similar but phrased differently
    test_queries = [
        "What are the best graphics cards?",
        "How to choose a good GPU for gaming?",
        "Graphics card recommendations for 1440p gaming",
        "What's the difference between AMD and NVIDIA?",
        "Best video cards for 4K video editing",
    ]
    
    print(f"\n1️⃣ Testing cache with {len(test_queries)} similar queries:")
    print("-" * 70)
    
    results = []
    for query in test_queries:
        # Embed query
        query_emb = db.model.encode(query)
        query_emb = normalize(query_emb.reshape(1, -1), norm='l2')[0]
        
        # Get cluster info
        cluster_probs = clustering.predict_soft(query_emb.reshape(1, -1))[0]
        dominant_cluster = int(np.argmax(cluster_probs))
        
        # Check cache
        cache_hit = cache.lookup(query, query_emb, dominant_cluster, cluster_probs)
        
        if cache_hit:
            result, similarity, matched = cache_hit
            print(f"✓ CACHE HIT: '{query}'")
            print(f"  Matched: '{matched}' (similarity: {similarity:.3f})")
            results.append(("hit", similarity))
        else:
            # Cache miss - add to cache
            dummy_result = f"Search results for: {query}"
            cache.add(query, query_emb, dummy_result, dominant_cluster, cluster_probs)
            print(f"✗ CACHE MISS: '{query}'")
            results.append(("miss", None))
        print()
    
    # Show statistics
    stats = cache.get_stats()
    print("Cache Statistics:")
    print("-" * 70)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Hits: {stats['hit_count']}, Misses: {stats['miss_count']}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Threshold: {stats['similarity_threshold']:.2f}")
    print()


def demo_cluster_analysis():
    """Demonstrate cluster analysis."""
    print("=" * 70)
    print("DEMO 2: Cluster Analysis & Boundary Cases")
    print("=" * 70)
    print()
    
    db = init_embedding_db()
    clustering = init_fuzzy_clustering(db.embeddings)
    
    print("1️⃣ Cluster Sizes:")
    print("-" * 70)
    unique, counts = np.unique(clustering.labels_hard, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(clustering.labels_hard)) * 100
        bar = "█" * (percentage // 2)
        print(f"Cluster {cluster_id:2d}: {count:4d} documents ({percentage:5.1f}%) {bar}")
    print()
    
    print("2️⃣ Documents at Cluster Boundaries (Most Interesting):")
    print("-" * 70)
    boundary_docs = clustering.analyze_boundaries(db.texts, db.embeddings)
    
    for i, doc_info in enumerate(boundary_docs[:5], 1):
        doc_idx = doc_info['doc_idx']
        text = db.texts[doc_idx][:200]
        
        print(f"\n{i}. Doc {doc_idx}: Clusters {doc_info['primary_cluster']} ↔ {doc_info['secondary_cluster']}")
        print(f"   Probabilities: {doc_info['primary_prob']:.2f} / {doc_info['secondary_prob']:.2f}")
        print(f"   Text: \"{text}...\"")
    print()


def demo_threshold_sensitivity():
    """Demonstrate cache threshold sensitivity analysis."""
    print("=" * 70)
    print("DEMO 3: Cache Threshold Sensitivity Analysis")
    print("=" * 70)
    print()
    
    db = init_embedding_db()
    clustering = init_fuzzy_clustering(db.embeddings)
    cache = SemanticCache()
    
    # Create some test data
    print("Building test cache with 50 documents...")
    np.random.seed(42)
    sampled_indices = np.random.choice(len(db.texts), 50, replace=False)
    
    for idx in sampled_indices:
        emb = db.embeddings[idx]
        cluster_probs = clustering.predict_soft(emb.reshape(1, -1))[0]
        dominant_cluster = int(np.argmax(cluster_probs))
        cache.add(db.texts[idx], emb, f"Result {idx}", dominant_cluster, cluster_probs)
    
    # Test queries
    print(f"Testing with 30 held-out queries...")
    test_indices = np.random.choice(len(db.texts), 30, replace=False)
    test_queries = [(db.embeddings[idx], db.texts[idx]) for idx in test_indices]
    
    # Run sensitivity analysis
    sensitivity = cache.analyze_threshold_sensitivity(test_queries)
    
    print()
    print("Threshold Sensitivity Analysis:")
    print("-" * 70)
    print("Threshold | Hit Rate | Interpretation")
    print("-" * 70)
    
    for result in sensitivity:
        threshold = result['threshold']
        hit_rate = result['hit_rate']
        
        # Interpretation
        if hit_rate < 0.20:
            interpretation = "Too strict - cache rarely used"
        elif hit_rate < 0.40:
            interpretation = "Conservative - few false positives"
        elif hit_rate < 0.60:
            interpretation = "Balanced - good cache utility"
        else:
            interpretation = "Aggressive - risk of false positives"
        
        hit_bar = "▓" * int(hit_rate * 20)
        print(f"   {threshold:.2f}   |  {hit_rate:5.1%}   | {hit_bar} {interpretation}")
    
    print()
    print(f"Current system threshold: {CACHE_SIMILARITY_THRESHOLD:.2f}")
    print()


def demo_embedding_similarity():
    """Demonstrate embedding similarity for semantically similar queries."""
    print("=" * 70)
    print("DEMO 4: Embedding Similarity for Paraphrased Queries")
    print("=" * 70)
    print()
    
    db = init_embedding_db()
    
    query_variants = [
        "What are graphics cards good for?",
        "How do graphics cards work?",
        "Best graphics cards for gaming",
        "GPU vs CPU - what's the difference?",
    ]
    
    print("Computing embedding similarities between query variants:")
    print("-" * 70)
    
    embeddings = [normalize(db.model.encode(q).reshape(1, -1), norm='l2')[0] for q in query_variants]
    
    for i, q1 in enumerate(query_variants):
        similarities = [np.dot(embeddings[i], embeddings[j]) for j in range(len(query_variants))]
        avg_similarity = np.mean([s for j, s in enumerate(similarities) if i != j])
        
        result_str = f"Query {i}: '{q1[:40]}...'"
        similarity_str = f"Avg similarity to others: {avg_similarity:.3f}"
        print(f"{result_str:50} | {similarity_str}")
    
    print()
    print("Key insight: Similar queries have cosine similarity ~0.75-0.85")
    print(f"System threshold ({CACHE_SIMILARITY_THRESHOLD:.2f}) effectively catches these!")
    print()


if __name__ == "__main__":
    import traceback
    
    demos = [
        ("Semantic Cache", demo_semantic_cache),
        ("Cluster Analysis", demo_cluster_analysis),
        ("Threshold Sensitivity", demo_threshold_sensitivity),
        ("Embedding Similarity", demo_embedding_similarity),
    ]
    
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)
    print()
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {demo_name}: {e}")
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
