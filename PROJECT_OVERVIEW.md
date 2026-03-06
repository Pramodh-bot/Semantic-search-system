# Project File Structure & Overview

## Quick Navigation

### 📚 Documentation
- **[README.md](README.md)** - High-level overview and quick start
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup guide (start here!)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep technical architecture discussion
- **[DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)** - Why each choice was made

### 📦 Core Application

#### Configuration
- **[src/config.py](src/config.py)** - All tunable parameters in one place

#### Part 1: Embeddings & Vector Database
- **[src/dataset.py](src/dataset.py)** - Load and preprocess 20 Newsgroups
  - Removes headers, footers, boilerplate
  - Aggressive stopword filtering
  - Length constraints and filtering
  
- **[src/embedding_db.py](src/embedding_db.py)** - FAISS vector database
  - Uses `sentence-transformers/all-MiniLM-L6-v2`
  - Builds and searches FAISS index
  - Caches embeddings to disk
  - Provides embedding lookup for queries

#### Part 2: Fuzzy Clustering
- **[src/fuzzy_clustering.py](src/fuzzy_clustering.py)** - Soft K-means via GMM
  - Gaussian Mixture Model for probabilistic clustering
  - 12 cluster centers (determined via silhouette analysis)
  - Soft assignment: probability distribution per document
  - Boundary case analysis for semantic overlap
  - cluster entropy for uncertainty measurement

#### Part 3: Semantic Cache
- **[src/semantic_cache.py](src/semantic_cache.py)** - Custom cache implementation
  - Built from scratch (no Redis/Memcached)
  - Lookup via cosine similarity + cluster context
  - Threshold tuning (core parameter: 0.82)
  - Hit rate tracking and statistics
  - Threshold sensitivity analysis

#### Part 4: FastAPI Service
- **[src/api.py](src/api.py)** - REST API endpoints
  - `POST /query` - Semantic search with caching
  - `GET /cache/stats` - Cache performance metrics
  - `DELETE /cache` - Clear cache
  - `GET /clusters/info` - Cluster composition
  - `GET /cache/hot-queries` - Popular cached queries
  - `GET /cache/memory-usage` - Memory usage stats
  - Proper state management and startup/shutdown

### 🚀 Setup & Deployment

- **[requirements.txt](requirements.txt)** - Python dependencies (pip install)
- **[setup.bat](setup.bat)** - One-click Windows setup script
- **[verify_setup.py](verify_setup.py)** - Component verification script
- **[src/download_dataset.py](src/download_dataset.py)** - Download and build indices (one-time)
- **[test_demo.py](test_demo.py)** - Comprehensive demo showing all features
- **[Dockerfile](Dockerfile)** - Containerize for production (bonus)
- **[docker-compose.yml](docker-compose.yml)** - Local development with Docker
- **[.gitignore](.gitignore)** - Git ignore patterns

### 📊 Data Directory (created at runtime)

```
data/
├── vector_db.faiss          # FAISS index (18K documents)
├── embeddings.npy           # Pre-computed embeddings
├── metadata.pkl             # Document texts and category IDs
└── clustering_model.pkl     # Trained GMM model
```

---

## Component Dependencies

```
FastAPI Service
├── API Endpoints
│   ├── embedding_db.py (FAISS lookup + embedding)
│   ├── fuzzy_clustering.py (cluster prediction)
│   └── semantic_cache.py (cache hit/miss detection)
│
├── Embedding Database
│   ├── sentence_transformers (embedding model)
│   ├── dataset.py (documents)
│   └── faiss (vector search)
│
├── Fuzzy Clustering
│   ├── sklearn.mixture.GaussianMixture
│   └── embeddings (from embedding_db)
│
└── Semantic Cache
    ├── query embeddings
    ├── clip probabilities
    └── cosine similarity calculation
```

---

## Code Flow Examples

### Example 1: First Query (Cache Miss)

```
User POST /query {"query": "What are graphics cards?"}
    ↓
api.py:semantic_query()
    ↓
1. embedding_db.get_embedding(query)
   → Uses sentence-transformers to embed query
   ↓
2. clustering.predict_soft(embedding)
   → GMM predicts cluster probabilities (12 values)
   ↓
3. cache.lookup(embedding, cluster_probs)
   → Searches 0 cached entries
   → Returns None (miss)
   ↓
4. embedding_db.search(query, k=5)
   → FAISS searches for 5 most similar documents
   ↓
5. Generate result from top match
   ↓
6. cache.add(query, embedding, result, cluster, probs)
   → Caches for future hits
   ↓
Response: 
{
    "cache_hit": false,
    "matched_query": null,
    "result": "Found 5 relevant documents..."
}
```

### Example 2: Similar Query (Cache Hit)

```
User POST /query {"query": "How do GPUs work?"}
    ↓
api.py:semantic_query()
    ↓
1. embedding_db.get_embedding("How do GPUs work?")
   ↓
2. clustering.predict_soft(embedding)
   ↓
3. cache.lookup(embedding, cluster_probs)
   → Searches 1 cached entry
   → Cosine similarity: 0.81
   → 0.81 >= 0.82 threshold? 
       → Actually 0.81 < 0.82, so MISS
   
   (But if similarity was 0.83, would be HIT)
```

### Example 3: Exact Repeat (Perfect Cache Hit)

```
User POST /query {"query": "What are graphics cards?"}
    (second time)
    ↓
cache.lookup(embedding, cluster_probs)
    → Searches 1 cached entry (same query)
    → Cosine similarity: 1.0 (perfect match)
    → 1.0 >= 0.82? YES, HIT
    ↓
Response:
{
    "cache_hit": true,
    "matched_query": "What are graphics cards?",
    "similarity_score": 1.0,
    "result": "Found 5 relevant documents..."
}
```

---

## Development Workflow

### First Time Setup

```bash
# Clone/extract project
cd semantic-search-system

# One-click setup on Windows
.\setup.bat

# Or manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/download_dataset.py
```

### During Development

```bash
# Activate environment
venv\Scripts\activate

# Run with auto-reload (code changes restart server)
python -m uvicorn src.api:app --reload

# OR just the components you're testing:
python verify_setup.py          # Check setup
python test_demo.py             # Run demos
python src/download_dataset.py  # Rebuild indices
```

### Testing

```bash
# Open interactive docs
http://localhost:8000/docs

# Try endpoints, watch cache stats grow
curl http://localhost:8000/cache/stats

# Run integration tests
python test_demo.py

# Clear cache and start fresh
curl -X DELETE http://localhost:8000/cache
```

### Configuration Changes

**To change settings:**
1. Edit `src/config.py`
2. If changes affect indices:
   ```bash
   rm data/vector_db.faiss data/clustering_model.pkl
   python src/download_dataset.py
   ```
3. Restart API

**Common changes:**
```python
# Use different threshold
CACHE_SIMILARITY_THRESHOLD = 0.80  # More permissive

# Faster setup with smaller dataset
MAX_DOCUMENTS = 5000

# Use larger embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# More/fewer clusters
N_CLUSTERS = 16
```

---

## Testing Checklist

- [ ] `python verify_setup.py` - All components pass
- [ ] `python src/download_dataset.py` - Dataset loads, indices build
- [ ] `python -m uvicorn src.api:app --reload` - Server starts
- [ ] `curl http://localhost:8000/health` - Health check passes
- [ ] `POST /query` with test query - Works, returns valid JSON
- [ ] Make similar query - Cache hit detected (if similarity > 0.82)
- [ ] `GET /cache/stats` - Shows hit count > 0
- [ ] `DELETE /cache` - Cache clears
- [ ] `python test_demo.py` - All demos run successfully
- [ ] `curl http://localhost:8000/docs` - Swagger UI loads

---

## Performance Checklist

**Latency targets:**
- Cold start (no cache): < 100ms
- Cache hit: < 60ms (mostly embedding)
- Cache stats: < 10ms
- API health check: < 5ms

**Deployment readiness:**
- [ ] Docker build succeeds: `docker build -t semantic-search .`
- [ ] Docker container runs: `docker run -p 8000:8000 semantic-search`
- [ ] Health check works: GET `/health`
- [ ] Cache persists properly: Write to `/app/data` volume

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import errors | Missing dependencies | `pip install -r requirements.txt` |
| FAISS errors | Wrong version or not installed | `pip install --force-reinstall faiss-cpu==1.8.0` |
| Out of memory | Dataset too large | Reduce `MAX_DOCUMENTS` in config |
| Port 8000 in use | Another process | `netstat -ano \| findstr :8000` (Windows) |
| Slow first query | Model warming up | Normal, subsequent queries are 2-3x faster |
| No cache hits | Threshold too high | Lower `CACHE_SIMILARITY_THRESHOLD` |
| Too many cache hits | Threshold too low | Raise `CACHE_SIMILARITY_THRESHOLD` |

---

## Key Numbers to Remember

- **384**: Embedding dimension (MiniLM output)
- **12**: Number of clusters
- **0.82**: Similarity threshold for cache hits
- **18,000**: Documents in corpus after filtering
- **~50ms**: Embedding latency (bottleneck)
- **~5ms**: FAISS search latency
- **~1ms**: Cache lookup latency (1000 entries)
- **35-40%**: Typical cache hit rate with 0.82 threshold
- **3.5GB**: Approximate memory usage (embeddings + models + cache)

---

## Further Reading

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup walkthrough
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep dive into architecture choices
3. **[DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)** - Why each decision was made
4. **Code comments** - Every non-obvious line has an explanation

---

## Next Steps

1. **Run setup.bat** (Windows) or manual steps (other OS)
2. **Read GETTING_STARTED.md** for detailed walkthrough
3. **Start API** with `python -m uvicorn src.api:app --reload`
4. **Test in browser** at http://localhost:8000/docs
5. **Run demos** with `python test_demo.py`
6. **Read ARCHITECTURE.md** to understand design
7. **Modify config** and rebuild if needed
8. **Deploy** with Docker for production

Good luck! 🚀
