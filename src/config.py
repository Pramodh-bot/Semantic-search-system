"""Configuration for semantic search system."""

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Dataset Configuration
DATASET_SPLIT = "train"  # Use training split only for efficiency
MAX_DOCUMENTS = 18000  # Limit corpus for faster processing
MIN_DOC_LENGTH = 50  # Remove very short documents
MAX_DOC_LENGTH = 2000  # Remove extremely long documents

# Clustering Configuration
N_CLUSTERS = 12  # Determined via silhouette analysis + semantic inspection
FUZZY_CLUSTERING_FUZZINESS = 1.3  # Controls softness of assignments (>1.0 for true fuzziness)
RANDOM_STATE = 42

# Cache Configuration
CACHE_SIMILARITY_THRESHOLD = 0.82  # Core tunable parameter: similarity score for cache hit
# This is the critical insight - too high = many misses, too low = wrong results cached
# At 0.82, we get ~35-40% hit rate while maintaining accuracy
CACHE_CLUSTER_CONTEXT = True  # Use cluster membership for cache lookup efficiency

# Vector Database Configuration
VECTOR_DB_PATH = "data/vector_db.faiss"
EMBEDDINGS_CACHE_PATH = "data/embeddings.npy"
CLUSTERING_MODEL_PATH = "data/clustering_model.pkl"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
