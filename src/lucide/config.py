"""Configuration for python-lucide.

This module contains default configuration values used throughout the package.
"""

# Default Lucide tag to use when building the icon database
DEFAULT_LUCIDE_TAG = "1.7.0"

# Default size for the LRU cache used by lucide_icon function
DEFAULT_ICON_CACHE_SIZE = 128

# --- Semantic search ---

# Text embedding model used at both build and query time (via fastembed)
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
DEFAULT_EMBEDDING_DIM = 768

# Nomic requires task-specific prefixes for asymmetric retrieval
EMBEDDING_QUERY_PREFIX = "search_query: "
EMBEDDING_DOCUMENT_PREFIX = "search_document: "

# VLM used to generate icon descriptions at build time
DEFAULT_VLM_MODEL = "gemini-2.5-flash-lite"

# URL template for downloading pre-built search data.
# {version} is replaced with the Lucide icon-set version (e.g. "0.577.0").
SEARCH_DB_URL_TEMPLATE = (
    "https://github.com/mmacpherson/python-lucide/releases/download/"
    "search-v{version}/lucide-search.db"
)
