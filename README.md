# Semantic Search System with Fuzzy Clustering & Custom Cache

A lightweight semantic search system built on the 20 Newsgroups dataset, featuring:
- **Fuzzy vector clustering** to uncover semantic structure beyond hard labels
- **Custom semantic cache** (no Redis/Memcached) using embedding similarity
- **FastAPI service** with live endpoints for semantic search and cache statistics

## Architecture

### Part 1: Embedding & Vector Database
- Uses `sentence-transformers` (MiniLM-L6-v2) for efficient embeddings
- FAISS vector database for sub-linear similarity search
- Data preprocessing: removal of headers/footers, aggressive stopword filtering, length constraints
- Rationale: News posts contain boilerplate; semantic content is in body text

### Part 2: Fuzzy Clustering
- Soft K-means clustering (not hard assignments)
- Each document gets a probability distribution over clusters
- Cluster count determined via silhouette analysis and semantic coherence
- Analysis includes cluster boundaries and inter-cluster relationships

### Part 3: Semantic Cache
- Custom implementation built from first principles
- Queries matched via cosine similarity + cluster context
- Threshold tuning is the core insight: what similarity threshold minimizes redundant computation?
- No external caching libraries

### Part 4: FastAPI Service
- `POST /query`: Semantic search with cache checking
- `GET /cache/stats`: Cache performance metrics  
- `DELETE /cache`: Clear cache and reset stats
- Proper state management with embeddings pre-loaded

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset and build indices (first run takes 5-10 minutes)
python src/download_dataset.py

# Start the API
uvicorn src.api:app --reload
```

Open `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
semantic-search-system/
├── src/
│   ├── api.py                 # FastAPI endpoints
│   ├── embedding_db.py        # FAISS vector database setup
│   ├── fuzzy_clustering.py    # Soft K-means implementation
│   ├── semantic_cache.py      # Custom cache layer
│   ├── dataset.py             # Dataset loading & preprocessing
│   └── download_dataset.py    # Initial data fetch and processing
├── data/                       # Cached embeddings, vectors, models
├── requirements.txt
└── README.md
```

## Key Design Decisions

1. **Embedding Model**: MiniLM-L6-v2 is 22M params, perfect for edge inference
2. **Vector Store**: FAISS over alternatives due to simplicity and speed
3. **Clustering**: Soft assignments (Gaussian mixture model style) to capture ambiguity
4. **Cache Key**: Cluster ID + embedding similarity, not just raw similarity
5. **Threshold**: Tunable parameter (~0.85 cosine similarity) determines cache hit rates

## Configuration

Edit `src/config.py` to adjust:
- Embedding model
- Number of clusters
- Cache similarity threshold
- Vector database parameters

## Testing

```bash
# Test API
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What are the best graphics cards?"}'

# Check cache stats
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

## Docker (Bonus)

```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```
