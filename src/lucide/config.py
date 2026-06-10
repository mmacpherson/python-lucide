"""Configuration for python-lucide.

This module contains default configuration values used throughout the package.
"""

from dataclasses import dataclass

# Default Lucide tag to use when building the icon database
DEFAULT_LUCIDE_TAG = "1.17.0"

# Default size for the LRU cache used by lucide_icon function
DEFAULT_ICON_CACHE_SIZE = 128

# --- Semantic search ---


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """One embedding model usable for icon search.

    Each model must be available both as a fastembed ONNX model (build time
    and Python query time) and as a transformers.js model (browser query
    time), and the two runtimes must produce identical vectors. ``pooling``
    is the transformers.js pooling mode that matches fastembed's output for
    this model — verified empirically (cosine 1.0) before a model is added
    here; mismatched pooling silently degrades search quality instead of
    erroring.

    Attributes:
        id: Short stable identifier stored in the search DB and web manifest.
        fastembed_model: fastembed model name.
        web_model: transformers.js (Hugging Face) model id. Empty means the
            model is Python-only and excluded from the web manifest.
        dim: Embedding dimensionality.
        pooling: transformers.js pooling mode ("mean" or "cls").
        web_dtype: transformers.js quantization. Document vectors always
            come from fastembed (fp32), so a quantized browser model only
            perturbs the query vector — q8 stays ~0.994 cosine to fp32,
            which is ranking-equivalent, and roughly quarters the download.
        query_prefix: Prefix prepended to queries (asymmetric retrieval).
        document_prefix: Prefix prepended to documents at build time.
        label: Human-facing label for the web UI model toggle.
    """

    id: str
    dim: int
    fastembed_model: str = ""
    web_model: str = ""
    pooling: str = "mean"
    web_dtype: str = "fp32"
    query_prefix: str = ""
    document_prefix: str = ""
    label: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    "minilm": EmbeddingModelConfig(
        id="minilm",
        fastembed_model="sentence-transformers/all-MiniLM-L6-v2",
        web_model="Xenova/all-MiniLM-L6-v2",
        dim=384,
        pooling="mean",
        label="Faster",
    ),
    "bge-small": EmbeddingModelConfig(
        id="bge-small",
        fastembed_model="BAAI/bge-small-en-v1.5",
        web_model="Xenova/bge-small-en-v1.5",
        dim=384,
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
        label="Better",
    ),
    "multilingual": EmbeddingModelConfig(
        id="multilingual",
        fastembed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        web_model="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        dim=384,
        pooling="mean",
        web_dtype="q8",
        label="Multilingual",
    ),
}

DEFAULT_SEARCH_MODEL_ID = "bge-small"

# VLM used to generate icon descriptions at build time
DEFAULT_VLM_MODEL = "gemini-2.5-flash-lite"

# Bumped when the search DB layout changes incompatibly (v2: multi-model
# embeddings keyed by (name, model)). Recorded in the DB metadata and baked
# into the local cache filename so a stale cache from an older package
# version is never misread after an upgrade.
SEARCH_DB_SCHEMA_VERSION = 2

# URL template for downloading pre-built search data.
# {version} is replaced with the Lucide icon-set version (e.g. "0.577.0").
SEARCH_DB_URL_TEMPLATE = (
    "https://github.com/mmacpherson/python-lucide/releases/download/"
    "search-v{version}/lucide-search.db"
)
