"""
Cluster interpretation using TF-IDF analysis.

Explains what each cluster represents by finding the top keywords
that distinguish that cluster from others.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from config import N_CLUSTERS


def interpret_clusters(documents: List[str], cluster_labels: np.ndarray, 
                      n_terms: int = 10, max_features: int = 5000) -> Dict[int, List[str]]:
    """
    Interpret clusters by finding top TF-IDF terms for each.
    
    Args:
        documents: List of document texts
        cluster_labels: Hard cluster assignment for each document (from dominant cluster)
        n_terms: Number of top terms to extract per cluster
        max_features: Maximum vocabulary size for TF-IDF
        
    Returns:
        Dictionary mapping cluster_id -> list of top terms
        
    Example:
        {
            0: ["gpu", "nvidia", "graphics", "cuda", "shader", ...],
            1: ["theology", "christian", "god", "faith", "bible", ...],
            ...
        }
    """
    print(f"Analyzing {len(np.unique(cluster_labels))} clusters using TF-IDF...")
    
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        lowercase=True,
        ngram_range=(1, 1)  # Only unigrams for clarity
    )
    
    # Fit on all documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    
    cluster_terms = {}
    
    # For each cluster, find top terms
    for cluster_id in range(N_CLUSTERS):
        # Get indices of documents in this cluster
        cluster_mask = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_mask) == 0:
            cluster_terms[cluster_id] = []
            continue
        
        # Get TF-IDF scores for this cluster's documents
        cluster_tfidf = tfidf_matrix[cluster_mask]
        
        # Compute mean TF-IDF per term for this cluster
        mean_tfidf = cluster_tfidf.mean(axis=0).A1
        
        # Find top n_terms
        top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
        top_terms = [terms[i] for i in top_indices if mean_tfidf[i] > 0]
        
        cluster_terms[cluster_id] = top_terms
    
    return cluster_terms


def print_cluster_interpretation(cluster_terms: Dict[int, List[str]]):
    """
    Pretty-print cluster interpretations.
    
    Example output:
        Cluster 0: gpu, nvidia, graphics, cuda, shader, processor, memory, speed, performance, display
        Cluster 1: theology, christian, god, faith, bible, religion, jesus, church, spiritual, gospel
        ...
    """
    for cluster_id in sorted(cluster_terms.keys()):
        terms = cluster_terms[cluster_id]
        if terms:
            print(f"Cluster {cluster_id}: {', '.join(terms)}")
        else:
            print(f"Cluster {cluster_id}: [no documents]")


def find_semantically_similar_clusters(cluster_terms: Dict[int, List[str]], 
                                       top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
    """
    Find which clusters have overlapping vocabularies.
    
    This helps understand topic relationships.
    
    Args:
        cluster_terms: Output from interpret_clusters()
        top_n: Number of similar clusters to return per cluster
        
    Returns:
        Dictionary mapping cluster_id -> [(similar_cluster_id, similarity_score), ...]
        
    Example:
        {
            0: [(5, 0.8), (3, 0.6)],  # Cluster 0 overlaps heavily with cluster 5
            ...
        }
    """
    similarities = {}
    
    for cluster_id in cluster_terms:
        terms_a = set(cluster_terms[cluster_id])
        
        # Skip if cluster has no terms
        if not terms_a:
            similarities[cluster_id] = []
            continue
        
        cluster_sims = []
        
        for other_id in cluster_terms:
            if cluster_id == other_id:
                continue
            
            terms_b = set(cluster_terms[other_id])
            
            # Jaccard similarity
            if len(terms_a | terms_b) == 0:
                similarity = 0
            else:
                similarity = len(terms_a & terms_b) / len(terms_a | terms_b)
            
            cluster_sims.append((other_id, similarity))
        
        # Sort by similarity and keep top N
        cluster_sims.sort(key=lambda x: x[1], reverse=True)
        similarities[cluster_id] = cluster_sims[:top_n]
    
    return similarities


def print_cluster_relationships(similarities: Dict[int, List[Tuple[int, float]]]):
    """Pretty-print cluster topic relationships."""
    print("\n=== Cluster Semantic Relationships ===")
    for cluster_id in sorted(similarities.keys()):
        related = similarities[cluster_id]
        if related:
            rel_str = ", ".join([f"Cluster {c}({s:.2f})" for c, s in related])
            print(f"Cluster {cluster_id} overlaps with: {rel_str}")
