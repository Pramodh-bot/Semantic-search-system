"""Fuzzy clustering using soft K-means (Gaussian Mixture Model)."""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import pickle
import os
from typing import Tuple, List
from config import N_CLUSTERS, RANDOM_STATE, CLUSTERING_MODEL_PATH, EMBEDDING_DIMENSION


class FuzzyClustering:
    """
    Soft clustering using Gaussian Mixture Models.
    
    Design rationale:
    - Soft assignments: Documents get probability distribution over clusters
    - GMM is theoretically sound: assumes multivariate normal per cluster
    - Probabilistic: uncertainty is explicit (important for cache precision)
    - Not K-means: Hard assignments lose semantic nuance
      Example: "gun legislation" belongs to both politics AND firearms clusters
    
    Cluster count determination:
    - We use silhouette score + semantic inspection
    - Too few clusters (n=5): Similar topics merged, boundary cases lost
    - Too many clusters (n=20): Fragmentation, no clear semantic boundaries
    - Sweet spot n=12: Balances granularity and coherence
    """
    
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.model = None
        self.cluster_centers = None
        self.labels_soft = None  # Soft assignments: shape (n_docs, n_clusters)
        self.labels_hard = None  # Hard assignments: argmax of soft
    
    def fit(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit GMM to embeddings.
        
        Args:
            embeddings: Shape (n_documents, embedding_dim), should be normalized
            
        Returns:
            Tuple of (soft_labels, hard_labels)
            - soft_labels: (n_docs, n_clusters) - probability of each cluster
            - hard_labels: (n_docs,) - highest probability cluster per doc
        """
        print(f"Training fuzzy clustering with {self.n_clusters} clusters...")
        
        # Fit Gaussian Mixture Model (probabilistic clustering)
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            random_state=RANDOM_STATE,
            max_iter=100,
            n_init=10,
        )
        
        self.model.fit(embeddings)
        
        # Get soft assignments (probability of each cluster)
        self.labels_soft = self.model.predict_proba(embeddings)
        
        # Get hard assignments (most likely cluster)
        self.labels_hard = self.model.predict(embeddings)
        
        self.cluster_centers = self.model.means_
        
        # Evaluate clustering quality
        silhouette = silhouette_score(embeddings, self.labels_hard)
        print(f"Silhouette score: {silhouette:.3f}")
        
        # Print cluster sizes
        unique, counts = np.unique(self.labels_hard, return_counts=True)
        print(f"Cluster sizes: {dict(zip(unique, counts))}")
        
        self._save()
        
        return self.labels_soft, self.labels_hard
    
    def predict_soft(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get soft cluster assignments for new embeddings.
        
        Args:
            embeddings: Shape (n_new, embedding_dim)
            
        Returns:
            Shape (n_new, n_clusters) - probability of each cluster
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict_proba(embeddings)
    
    def predict_hard(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get hard cluster assignments for new embeddings.
        
        Args:
            embeddings: Shape (n_new, embedding_dim)
            
        Returns:
            Shape (n_new,) - cluster ID for each document
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(embeddings)
    
    def get_cluster_entropy(self, doc_idx: int) -> float:
        """
        Calculate entropy of cluster distribution for a document.
        Shows how uncertain the model is about cluster membership.
        
        Low entropy (0.1) = high confidence in one cluster
        High entropy (log(n_clusters)) = uniform uncertainty across all
        
        Useful for cache: high uncertainty documents are more likely mismatches
        """
        from scipy.stats import entropy
        return entropy(self.labels_soft[doc_idx])
    
    def get_cluster_info(self, cluster_id: int, embeddings: np.ndarray) -> dict:
        """
        Get semantic information about a cluster.
        
        Args:
            cluster_id: Cluster index
            embeddings: Original embeddings to find most representative docs
            
        Returns:
            Dictionary with cluster statistics
        """
        # Documents assigned to this cluster
        mask = self.labels_hard == cluster_id
        n_documents = mask.sum()
        
        # Soft assignment statistics
        soft_probs = self.labels_soft[:, cluster_id]
        avg_membership = soft_probs[mask].mean()
        
        # Find most representative documents (highest probability)
        top_indices = np.argsort(soft_probs)[-5:][::-1]
        
        return {
            'cluster_id': cluster_id,
            'num_documents': int(n_documents),
            'avg_membership_strength': float(avg_membership),
            'top_member_indices': top_indices.tolist(),
            'center': self.cluster_centers[cluster_id].tolist(),
        }
    
    def analyze_boundaries(self, texts: List[str], embeddings: np.ndarray) -> List[dict]:
        """
        Analyze documents at cluster boundaries.
        
        These are the most interesting - they show semantic overlap between clusters.
        A document at the boundary between politics and firearms clusters might be
        about gun legislation.
        
        Returns list of {doc_idx, primary_cluster, secondary_cluster, probabilities}
        """
        # Find documents with high uncertainty across 2+ clusters
        max_prob = self.labels_soft.max(axis=1)
        
        # Documents where top 2 clusters have similar probabilities
        results = []
        for doc_idx in range(len(texts)):
            sorted_probs = np.sort(self.labels_soft[doc_idx])[::-1]
            
            # If gap between 1st and 2nd is small, it's a boundary case
            if sorted_probs[0] - sorted_probs[1] < 0.15:
                top_clusters = np.argsort(self.labels_soft[doc_idx])[-2:][::-1]
                results.append({
                    'doc_idx': doc_idx,
                    'primary_cluster': int(top_clusters[0]),
                    'secondary_cluster': int(top_clusters[1]),
                    'primary_prob': float(self.labels_soft[doc_idx][top_clusters[0]]),
                    'secondary_prob': float(self.labels_soft[doc_idx][top_clusters[1]]),
                })
        
        # Sort by probability difference (smallest = most ambiguous)
        results.sort(key=lambda x: abs(x['primary_prob'] - x['secondary_prob']))
        
        return results[:20]  # Return top 20 most ambiguous documents
    
    def _save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(CLUSTERING_MODEL_PATH) or '.', exist_ok=True)
        
        state = {
            'model': self.model,
            'labels_soft': self.labels_soft,
            'labels_hard': self.labels_hard,
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
        }
        
        with open(CLUSTERING_MODEL_PATH, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Saved clustering model to {CLUSTERING_MODEL_PATH}")
    
    def load(self) -> bool:
        """Load model from disk."""
        if not os.path.exists(CLUSTERING_MODEL_PATH):
            return False
        
        with open(CLUSTERING_MODEL_PATH, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.labels_soft = state['labels_soft']
        self.labels_hard = state['labels_hard']
        self.cluster_centers = state['cluster_centers']
        self.n_clusters = state['n_clusters']
        
        print(f"Loaded clustering model with {self.n_clusters} clusters")
        return True


def init_fuzzy_clustering(embeddings: np.ndarray) -> FuzzyClustering:
    """Initialize clustering, training if necessary."""
    clustering = FuzzyClustering()
    
    if clustering.load():
        return clustering
    
    clustering.fit(embeddings)
    return clustering


if __name__ == "__main__":
    from embedding_db import init_embedding_db
    
    db = init_embedding_db()
    clustering = init_fuzzy_clustering(db.embeddings)
    
    # Analyze boundary cases
    print("\nMost ambiguous documents at cluster boundaries:")
    boundary_docs = clustering.analyze_boundaries(db.texts, db.embeddings)
    
    for doc_info in boundary_docs[:5]:
        doc_idx = doc_info['doc_idx']
        print(f"\nDoc {doc_idx}:")
        print(f"  Primary cluster: {doc_info['primary_cluster']} (prob: {doc_info['primary_prob']:.3f})")
        print(f"  Secondary cluster: {doc_info['secondary_cluster']} (prob: {doc_info['secondary_prob']:.3f})")
        print(f"  Text: {db.texts[doc_idx][:150]}...")
