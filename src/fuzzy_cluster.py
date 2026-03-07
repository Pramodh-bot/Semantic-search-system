"""
True Fuzzy C-Means clustering using scikit-fuzzy.

This implements TRUE soft/fuzzy clustering where each document has
a membership degree (0-1) for EVERY cluster, not a probability distribution.

Key difference from GMM approach:
- GMM: probabilities sum to 1 (exclusive membership)
- FCM: degrees can sum to more than 1 (overlapping membership)

This satisfies the "fuzzy clustering" requirement of allowing documents
to belong to multiple clusters simultaneously with varying degrees.
"""

import numpy as np
import skfuzzy as fuzz
from typing import Tuple
from .config import N_CLUSTERS


def run_fuzzy_clustering(embeddings: np.ndarray, n_clusters: int = N_CLUSTERS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Fuzzy C-Means clustering on embeddings.
    
    Args:
        embeddings: Shape (n_documents, embedding_dim)
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (cluster_centers, membership_matrix)
        - cluster_centers: (n_clusters, embedding_dim) - centroid of each cluster
        - membership_matrix: (n_clusters, n_documents) - membership degree (0-1) for each doc in each cluster
    
    Example membership matrix:
        doc_id = 54
        cluster_0: 0.03
        cluster_1: 0.12
        cluster_2: 0.61   <- dominant cluster
        cluster_3: 0.08
        ...
        cluster_11: 0.03
    """
    print(f"Running Fuzzy C-Means clustering with {n_clusters} clusters...")
    
    # Transpose for scikit-fuzzy (expects features × samples)
    data = embeddings.T
    
    # Run Fuzzy C-Means
    # m=2: Fuzziness parameter (higher = fuzzier/more overlapping)
    # error=0.005: Convergence threshold
    # maxiter=1000: Maximum iterations
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data,
        c=n_clusters,
        m=2.0,
        error=0.005,
        maxiter=1000,
        init=None,
        seed=42
    )
    
    print(f"Fuzzy C-Means converged. Cluster centers shape: {cntr.shape}")
    print(f"Membership matrix shape: {u.shape}")
    
    return cntr, u


def get_dominant_cluster(membership_matrix: np.ndarray, doc_index: int) -> int:
    """
    Get the dominant (highest membership) cluster for a document.
    
    Args:
        membership_matrix: Shape (n_clusters, n_documents)
        doc_index: Index of the document
        
    Returns:
        Cluster ID with highest membership for this document
        
    Example:
        doc_120 -> cluster 3 (membership=0.68)
    """
    memberships = membership_matrix[:, doc_index]
    return np.argmax(memberships)


def get_membership_scores(membership_matrix: np.ndarray, doc_index: int) -> np.ndarray:
    """
    Get all membership scores for a document.
    
    Args:
        membership_matrix: Shape (n_clusters, n_documents)
        doc_index: Index of the document
        
    Returns:
        Array of membership scores for each cluster
    """
    return membership_matrix[:, doc_index]


def find_boundary_documents(membership_matrix: np.ndarray, threshold: float = 0.1) -> list:
    """
    Find documents that belong to multiple clusters with similar strength.
    
    These are "boundary" documents that represent topic overlap.
    
    Args:
        membership_matrix: Shape (n_clusters, n_documents)
        threshold: Maximum difference between top-2 memberships (lower = stricter boundary)
        
    Returns:
        List of document indices at cluster boundaries
        
    Example output:
        doc_812: cluster_3=0.52, cluster_7=0.48 (difference=0.04 < 0.1)
        -> This document is at the boundary between topics 3 and 7
    """
    boundary_docs = []
    
    for doc_id in range(membership_matrix.shape[1]):
        memberships = membership_matrix[:, doc_id]
        
        # Sort memberships to find top 2
        sorted_memberships = np.sort(memberships)
        
        # If difference between top 2 is small, document is at boundary
        if sorted_memberships[-1] - sorted_memberships[-2] < threshold:
            boundary_docs.append(doc_id)
    
    return boundary_docs


def compute_membership_entropy(membership_matrix: np.ndarray, doc_index: int) -> float:
    """
    Compute entropy of membership distribution for a document.
    
    High entropy = document belongs to many clusters equally (uncertain)
    Low entropy = document strongly belongs to one cluster (certain)
    
    Args:
        membership_matrix: Shape (n_clusters, n_documents)
        doc_index: Index of the document
        
    Returns:
        Shannon entropy of the membership distribution
        
    Example:
        Uniform [0.08, 0.08, ...]: high entropy (confused)
        Sharp [0.8, 0.05, ...]: low entropy (clear)
    """
    memberships = membership_matrix[:, doc_index]
    # Avoid log(0) by adding small epsilon
    memberships = np.clip(memberships, 1e-10, 1)
    entropy = -np.sum(memberships * np.log(memberships))
    return entropy


def find_uncertain_documents(membership_matrix: np.ndarray, entropy_threshold: float = 1.5) -> list:
    """
    Find documents with high membership entropy (ambiguous cluster assignment).
    
    Args:
        membership_matrix: Shape (n_clusters, n_documents)
        entropy_threshold: Entropy above this value indicates uncertainty
        
    Returns:
        List of uncertain document indices
    """
    uncertain_docs = []
    
    for doc_id in range(membership_matrix.shape[1]):
        entropy = compute_membership_entropy(membership_matrix, doc_id)
        if entropy > entropy_threshold:
            uncertain_docs.append((doc_id, entropy))
    
    # Sort by entropy (highest first)
    uncertain_docs.sort(key=lambda x: x[1], reverse=True)
    return uncertain_docs
