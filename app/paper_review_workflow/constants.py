"""Constants for paper review workflow."""

# Ranking related
DEFAULT_TOP_N_PAPERS = 20  # Default number of top N papers
MAX_DISPLAY_PAPERS = 20    # Maximum number of papers to display

# LLM evaluation related
DEFAULT_LLM_MAX_TOKENS = 1000      # Default maximum tokens for LLM evaluation
DEFAULT_LLM_TEMPERATURE = 0.0      # Default temperature for LLM evaluation
DEFAULT_LLM_TIMEOUT = 60           # Default timeout for LLM evaluation (seconds)
PRELIMINARY_LLM_MAX_TOKENS = 50    # Maximum tokens for preliminary LLM evaluation

# Text processing related
ABSTRACT_SHORT_LENGTH = 300        # Character count for shortened abstract
MAX_KEYWORDS_DISPLAY = 8           # Maximum number of keywords to display
MAX_AUTHORS_DISPLAY = 5            # Maximum number of authors to display
MAX_RATIONALE_LENGTH = 500         # Maximum character count for evaluation rationale

# Synonym generation related
SYNONYMS_LLM_MAX_TOKENS = 200      # Maximum tokens for synonym generation
SYNONYMS_COUNT_MIN = 3             # Minimum number of synonyms
SYNONYMS_COUNT_MAX = 5             # Maximum number of synonyms

# Cache related
DEFAULT_CACHE_TTL_HOURS = 24       # Default TTL for cache (hours)
CACHE_DIR_NAME = "storage/cache"   # Cache directory name

# Scoring related
MIN_SCORE = 0.0                    # Minimum score value
MAX_SCORE = 1.0                    # Maximum score value
NEURIPS_RATING_SCALE = 10.0        # NeurIPS rating scale

# Weights for relevance score calculation
RELEVANCE_KEYWORD_WEIGHT = 0.70    # Weight for paper keyword match
RELEVANCE_TEXT_WEIGHT = 0.20       # Weight for title/abstract match
RELEVANCE_COVERAGE_WEIGHT = 0.10   # Weight for coverage

