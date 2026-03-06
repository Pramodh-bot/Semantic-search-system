"""FastAPI service for semantic search with caching."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
from sklearn.preprocessing import normalize
import time

# Import our components
from embedding_db import init_embedding_db, EmbeddingDatabase
from fuzzy_clustering import init_fuzzy_clustering, FuzzyClustering
from semantic_cache import get_cache, SemanticCache
from config import API_HOST, API_PORT


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request body for semantic search."""
    query: str = Field(..., min_length=1, max_length=5000, description="User's search query")


class QueryResponse(BaseModel):
    """Response from semantic search endpoint."""
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int
    cluster_probabilities: list


class CacheStats(BaseModel):
    """Cache statistics response."""
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    similarity_threshold: float = Field(description="Current cache similarity threshold")


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Holds application state."""
    def __init__(self):
        self.embedding_db: Optional[EmbeddingDatabase] = None
        self.clustering: Optional[FuzzyClustering] = None
        self.cache: Optional[SemanticCache] = None
        self.ready = False


app_state = AppState()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Semantic Search System",
    description="20 Newsgroups semantic search with fuzzy clustering and custom caching",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """
    Initialize all components on startup.
    
    This loads the embedding database, clustering model, and cache.
    Takes ~5-10 seconds depending on machine.
    """
    print("🚀 Starting up semantic search system...")
    
    try:
        print("  - Loading embedding database...")
        app_state.embedding_db = init_embedding_db()
        
        print("  - Loading fuzzy clustering model...")
        app_state.clustering = init_fuzzy_clustering(app_state.embedding_db.embeddings)
        
        print("  - Initializing semantic cache...")
        app_state.cache = get_cache()
        
        app_state.ready = True
        print("✅ System ready!")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down semantic search system...")
    app_state.ready = False


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Check if system is ready."""
    return {
        "status": "ready" if app_state.ready else "initializing",
        "embedding_db": app_state.embedding_db is not None,
        "clustering": app_state.clustering is not None,
        "cache": app_state.cache is not None,
    }


# ============================================================================
# Core Endpoints
# ============================================================================

@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def semantic_query(request: QueryRequest) -> QueryResponse:
    """
    Semantic search with cache checking.
    
    The flow:
    1. Embed the query using the same model as the corpus
    2. Get soft cluster assignments to understand semantic context
    3. Check semantic cache for similar previous queries
    4. If cache hit: return cached result
    5. If cache miss: search corpus, compute result, cache it
    
    The result includes:
    - dominant_cluster: Most likely semantic category
    - cluster_probabilities: Full distribution (shows confidence)
    - cache_hit: Whether this was retrieved from cache
    - similarity_score: How similar to cached query (if hit)
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    query_text = request.query.strip()
    
    # Embed query
    query_embedding = app_state.embedding_db.get_embedding(query_text)
    query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')[0]
    
    # Get cluster assignments for this query
    cluster_probs = app_state.clustering.predict_soft(query_embedding.reshape(1, -1))[0]
    dominant_cluster = int(np.argmax(cluster_probs))
    
    # Check cache
    cache_result = app_state.cache.lookup(
        query_text,
        query_embedding,
        dominant_cluster,
        cluster_probs
    )
    
    if cache_result:
        # Cache hit
        result_text, similarity, matched_query = cache_result
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=matched_query,
            similarity_score=float(similarity),
            result=result_text,
            dominant_cluster=dominant_cluster,
            cluster_probabilities=cluster_probs.tolist(),
        )
    
    # Cache miss - search corpus
    docs, similarities, doc_indices = app_state.embedding_db.search(query_text, k=5)
    
    # Generate result from top match
    if docs:
        # Summary is just first 500 chars of top match + metadata
        result_text = (
            f"[Top match similarity: {similarities[0]:.3f}]\n\n"
            f"Found {len(doc_indices)} relevant documents:\n"
            f"1. {docs[0][:500]}...\n\n"
            f"(Run with cache_hit=true to see full context)"
        )
    else:
        result_text = "No relevant documents found."
    
    # Cache the result
    app_state.cache.add(
        query_text,
        query_embedding,
        result_text,
        dominant_cluster,
        cluster_probs
    )
    
    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_text,
        dominant_cluster=dominant_cluster,
        cluster_probabilities=cluster_probs.tolist(),
    )


@app.get("/cache/stats", response_model=CacheStats, tags=["Cache"])
async def get_cache_stats() -> CacheStats:
    """
    Get current cache statistics.
    
    Metrics provided:
    - total_entries: Number of cached queries
    - hit_count: How many times something was retrieved from cache
    - miss_count: How many times we had to search the corpus
    - hit_rate: hit_count / (hit_count + miss_count)
    - similarity_threshold: Current threshold for cache hits (tunable)
    
    Interpretation:
    - hit_rate ~35-40% is typical for well-tuned threshold (0.82)
    - hit_rate >50% usually means threshold too low (false positives)
    - hit_rate <20% usually means threshold too high (unused cache)
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    stats = app_state.cache.get_stats()
    return CacheStats(**stats)


@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """
    Clear all cached entries and reset statistics.
    
    Use this to start fresh with a different threshold or to reset metrics.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    app_state.cache.clear()
    
    return {
        "status": "success",
        "message": "Cache cleared",
        "timestamp": time.time(),
    }


# ============================================================================
# Bonus: Analysis Endpoints
# ============================================================================

@app.get("/clusters/analysis", tags=["Analysis"])
async def get_cluster_analysis():
    """
    CRITICAL FOR ASSIGNMENT: Get interpretation of what each cluster represents.
    
    Returns:
    - cluster_id: Cluster number (0-11)
    - size: How many documents in this cluster
    - percentage: Percentage of corpus
    - avg_membership_strength: How strongly documents belong to this cluster
    - coherence: How similar documents are within the cluster (0-1)
    - top_representative_docs: Documents that best represent this cluster
    
    This shows the semantic meaning of each cluster.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    interpretations = app_state.clustering.interpret_clusters(
        app_state.embedding_db.texts,
        app_state.embedding_db.embeddings
    )
    
    return {
        "cluster_interpretations": interpretations,
        "total_clusters": app_state.clustering.n_clusters,
        "total_documents": len(app_state.embedding_db),
    }


@app.get("/clusters/boundaries", tags=["Analysis"])
async def get_boundary_documents():
    """
    CRITICAL FOR ASSIGNMENT: Get documents at semantic cluster boundaries.
    
    These documents have HIGH UNCERTAINTY - they could belong to multiple clusters.
    Example: "gun legislation" belongs to both politics (0.52) and firearms (0.48)
    
    Returns documents sorted by uncertainty (most ambiguous first).
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    boundaries = app_state.clustering.analyze_boundaries(
        app_state.embedding_db.texts,
        app_state.embedding_db.embeddings
    )
    
    return {
        "boundary_documents": boundaries,
        "interpretation": "These documents sit at the boundary between clusters. "
                        "They are interesting because they show where semantic categories overlap.",
    }


@app.get("/clusters/uncertainty", tags=["Analysis"])
async def get_uncertain_documents():
    """
    CRITICAL FOR ASSIGNMENT: Get most uncertain documents.
    
    Uncertainty is measured by entropy across cluster probabilities.
    High entropy = model doesn't know which cluster this belongs to.
    
    These documents are exactly the ones that reveal the true complexity of the data.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    uncertain = app_state.clustering.analyze_uncertainty(app_state.embedding_db.texts)
    
    return {
        "uncertain_documents": uncertain,
        "note": "uncertainty_ratio of 0.5 means completely uncertain (max entropy = 0.5 * log(12))",
    }


@app.get("/cache/hot-queries", tags=["Analysis"])
async def get_hot_queries(top_k: int = 10):
    """
    Get most frequently accessed cached queries.
    
    Tells you what people are asking about repeatedly.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    hot = app_state.cache.get_hot_queries(top_k)
    return {"hot_queries": hot}


@app.get("/cache/memory-usage", tags=["Analysis"])
async def get_memory_usage():
    """
    Estimate memory usage of current cache.
    
    Shows cache efficiency in terms of storage.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    memory_mb = app_state.cache.estimate_memory_usage()
    
    return {
        "cache_memory_mb": round(memory_mb, 2),
        "cache_entries": len(app_state.cache.entries),
        "bytes_per_entry": round((memory_mb * 1024 * 1024) / max(len(app_state.cache.entries), 1), 0),
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting server on {API_HOST}:{API_PORT}")
    print("Open http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
