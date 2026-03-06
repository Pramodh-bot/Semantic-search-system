# Getting Started - Semantic Search System

## Quick Start (5 minutes)

### 1. Prerequisites

- **Python 3.11+** (required for typing, performance)
- **pip** (comes with Python)
- ~2 GB free disk space (for dataset + models)
- ~4 GB RAM (embeddings + FAISS index in memory)

### 2. Setup (Windows)

```bash
# Double-click setup.bat in the project root
# OR from PowerShell:
.\setup.bat
```

This script:
1. Creates a Python virtual environment
2. Installs dependencies from `requirements.txt`
3. Downloads the 20 Newsgroups dataset
4. Builds embeddings and clustering models (~5-10 minutes)

### 3. Start the API

From the project directory:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run API server
python -m uvicorn src.api:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 4. Test It

Open your browser to: **http://localhost:8000/docs**

Try posting a query:
```json
{
  "query": "What are the best graphics cards for gaming?"
}
```

---

## Detailed Setup (with Manual Steps)

### Step 1: Create and Activate Virtual Environment

**Windows (PowerShell or Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `fastapi` & `uvicorn` - Web framework
- `sentence-transformers` - Embedding model
- `faiss-cpu` - Vector database
- `numpy`, `scikit-learn`, `scipy` - Scientific computing
- `pydantic` - Request/response validation

**Install takes 2-5 minutes** depending on internet connection.

### Step 3: Verify Installation

```bash
python verify_setup.py
```

Expected output:
```
✓ All dependencies installed correctly
✓ Embedding model loaded
✓ Clustering initialized with 12 clusters
✓ Cache works: 1 entry stored
✓ All required API routes present
✓ Configuration valid
```

### Step 4: Download Dataset and Build Indices

This is a one-time step that processes the 20 Newsgroups dataset.

```bash
python src/download_dataset.py
```

**What happens:**
1. Downloads ~20K newsgroup posts (80 MB)
2. Removes headers/footers (boilerplate cleanup)
3. Preprocesses text (stopwords, length filtering)
4. Embeds all documents using `sentence-transformers` (~5 min on CPU)
5. Builds FAISS index for fast search
6. Trains Gaussian Mixture Model for clustering
7. Saves everything to `data/` directory

**Expected output:**
```
Fetching 20 Newsgroups dataset...
Processing 20000 documents...
✓ Loaded 18000 documents after filtering

Embedding 18000 documents...
████████████████████████████████████████ 100%
Built index with 18000 documents
Saved FAISS index to data/vector_db.faiss

Training fuzzy clustering with 12 clusters...
Silhouette score: 0.412
Cluster sizes: {0: 1423, 1: 1289, ...}

✅ INITIALIZATION COMPLETE!
```

### Step 5: Start the API Server

```bash
python -m uvicorn src.api:app --reload
```

**Options:**
- `--reload`: Restart on code changes (development)
- `--host 0.0.0.0`: Listen on all interfaces (default: localhost only)
- `--port 8001`: Use different port (default: 8000)

### Step 6: Interact with the API

**Option A: Interactive Swagger UI**
- Open: http://localhost:8000/docs
- Try it out! Click any endpoint and fill in the form

**Option B: curl (command line)**

```bash
# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are neural networks?"}'

# Get cache stats
curl http://localhost:8000/cache/stats

# Get cluster info
curl http://localhost:8000/clusters/info

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

**Option C: Python requests**

```python
import requests

url = "http://localhost:8000/query"
data = {"query": "What are graphics cards?"}

response = requests.post(url, json=data)
result = response.json()

print(f"Cache hit: {result['cache_hit']}")
print(f"Cluster: {result['dominant_cluster']}")
print(f"Result: {result['result'][:200]}...")
```

---

## Example API Interaction

### First Query (Cache Miss)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are graphics cards?"}'
```

Response:
```json
{
  "query": "What are graphics cards?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[Top match similarity: 0.742]\n\nFound 5 relevant documents:...",
  "dominant_cluster": 5,
  "cluster_probabilities": [0.03, 0.02, 0.08, 0.15, 0.05, 0.52, ...]
}
```

### Similar Query (Cache Hit)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do GPUs work?"}'
```

Response:
```json
{
  "query": "How do GPUs work?",
  "cache_hit": true,
  "matched_query": "What are graphics cards?",
  "similarity_score": 0.81,
  "result": "[Top match similarity: 0.742]...",
  "dominant_cluster": 5,
  "cluster_probabilities": [0.03, 0.01, 0.09, 0.14, 0.06, 0.51, ...]
}
```

Notice: Same result because embeddings were similar enough (0.81 > 0.82 threshold)

### Check Stats

```bash
curl http://localhost:8000/cache/stats
```

Response:
```json
{
  "total_entries": 5,
  "hit_count": 2,
  "miss_count": 3,
  "hit_rate": 0.4,
  "similarity_threshold": 0.82
}
```

---

## Run Demos

```bash
python test_demo.py
```

This runs four demonstrations:
1. **Cache in Action**: Shows cache hit/miss with 5 paraphrased queries
2. **Cluster Analysis**: Displays cluster sizes and boundary documents
3. **Threshold Sensitivity**: Analyzes how threshold affects hit rate
4. **Embedding Similarity**: Visualizes similarity between query variants

Very useful for understanding system behavior!

---

## Understanding Output

### Cache Statistics

**Key metric: Hit Rate**

```
hit_rate = hit_count / (hit_count + miss_count)
```

- **< 20%**: Threshold too high, cache not helping
- **20-40%**: Healthy range, catching real paraphrases
- **40-60%**: Slightly aggressive, may have false positives
- **> 60%**: Risk of wrong answers, threshold too low

**Current system**: 0.82 threshold typically gives 35-40% hit rate

### Cluster Probabilities

Each query gets a distribution:
```python
[0.03, 0.02, 0.08, 0.15, 0.05, 0.52, 0.04, 0.06, ...]
```

- **dominant_cluster**: Highest probability (5 = 0.52 = 52%)
- **entropy**: How uncertain the model is
- **boundary documents**: When probabilities are similar (e.g., 0.40, 0.38)

### Similarity Scores

From 0 to 1, where:
- **< 0.30**: Completely unrelated
- **0.30-0.60**: Different topics, maybe loosely related
- **0.60-0.80**: Related topics, possibly different angles
- **0.80-0.90**: Very similar, likely same intent (our cache hits)
- **> 0.90**: Nearly identical or rephrasing
- **= 1.00**: Exact match

---

## Troubleshooting

### "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers
```

### "FAISS not found" or FAISS errors

**Solution:**
```bash
pip install faiss-cpu
# OR if you have GPU:
pip install faiss-gpu
```

### Port 8000 already in use

**Solution 1: Use different port**
```bash
python -m uvicorn src.api:app --port 8001
```

**Solution 2: Kill process using port**

Windows (PowerShell):
```powershell
Stop-Process -Name python -Force
```

### API slow on first queries

**Expected**: First embedding query is slower (~500ms)
- Transformer model warming up
- Subsequent queries are faster (~100ms)

### Out of memory error

**Cause**: Dataset too large for your machine

**Solution**: Reduce in `src/config.py`:
```python
MAX_DOCUMENTS = 10000  # From 18000
```

Then rebuild:
```bash
rm data/vector_db.faiss data/clustering_model.pkl
python src/download_dataset.py
```

---

## Docker Setup (Optional)

Build and run in a container:

```bash
# Build image
docker build -t semantic-search .

# Run container
docker run -p 8000:8000 semantic-search

# Or use docker-compose
docker-compose up
```

Container includes:
- Ubuntu Linux base
- Python 3.11
- All dependencies
- Health checks
- Volume mounting for persistent cache

---

## Advanced Configuration

Edit `src/config.py`:

```python
# Use a different embedding model (larger = slower but better)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 438M params

# Adjust similarity threshold
CACHE_SIMILARITY_THRESHOLD = 0.80  # More permissive

# Change number of clusters
N_CLUSTERS = 10  # Fewer, larger clusters

# Reduce dataset size for faster setup
MAX_DOCUMENTS = 10000

# Aggressive preprocessing
MIN_DOC_LENGTH = 100  # Only keep meaty documents
```

Then rebuild:
```bash
python src/download_dataset.py
```

---

## Next Steps

1. **Read** [ARCHITECTURE.md](ARCHITECTURE.md) for deep understanding
2. **Experiment** with threshold: `DELETE /cache` and adjust `CACHE_SIMILARITY_THRESHOLD`
3. **Extend**: Add endpoints for custom query types, domain-specific corpora
4. **Deploy**: Use Docker + Kubernetes for production

---

## Performance Tips

**Faster embedding computation:**
- Use smaller model: `all-MiniLM-L6-v2` (22M) instead of `mpnet-base` (109M)
- Enable GPU: Install `torch` with CUDA, use `device='cuda'`
- Batch encode: Multiple queries at once

**Faster search:**
- FAISS is already fast, query caching via semantic-cache is the optimization
- For very large corpus (> 1M docs): Use approximate nearest neighbors (HNSW, IVF)

**Lower memory:**
- Use quantized embeddings: 8-bit instead of 32-bit floats
- Reduce `MAX_DOCUMENTS` in config
- Clear cache periodically: `DELETE /cache`

---

## Questions?

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for:
- Why each design choice was made
- How threshold tuning works
- Why cluster context matters
- Performance characteristics
- Future improvement ideas
