"""Configuration for PaperReviewAgent."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMModel(str, Enum):
    """Supported LLM models (OpenAI GPT models only)."""
    
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    GPT4_TURBO = "gpt-4-turbo"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"


class LLMConfig:
    """Configuration for LLM evaluation."""
    
    def __init__(
        self,
        model: LLMModel = LLMModel.GPT4O_MINI,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 60,
        max_concurrent: int = 10,
    ):
        """Initialize LLMConfig.
        
        Args:
        ----
            model: LLM model to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum number of tokens
            timeout: Timeout in seconds
            max_concurrent: Maximum concurrent executions (parallel processing)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_concurrent = max_concurrent
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.value,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_concurrent": self.max_concurrent,
        }


@dataclass
class ScoringWeights:
    """Scoring weight configuration."""
    
    # Integration weights for OpenReview and LLM
    openreview_weight: float = 0.4  # Weight for OpenReview evaluation
    llm_weight: float = 0.6          # Weight for LLM evaluation
    
    # Score weights within OpenReview (weights for relevance, novelty, impact)
    relevance_weight: float = 0.4   # Weight for relevance
    novelty_weight: float = 0.3     # Weight for novelty
    impact_weight: float = 0.3      # Weight for impact
    
    # Weights for relevance score calculation (optimized for synonym expansion)
    keyword_exact_match_weight: float = 0.3   # Weight for exact match
    keyword_partial_match_weight: float = 0.15  # Weight for partial match (increased from 0.1 to 0.15)
    review_quality_weight: float = 0.25        # [Unused] Weight for review quality (excluded as it's independent of relevance)
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        # Check sum of OpenReview and LLM weights
        total_integration = self.openreview_weight + self.llm_weight
        if abs(total_integration - 1.0) > 0.01:
            raise ValueError(
                f"openreview_weight + llm_weight must equal 1.0 (got {total_integration})"
            )
        
        # Check sum of score weights within OpenReview
        total_openreview = self.relevance_weight + self.novelty_weight + self.impact_weight
        if abs(total_openreview - 1.0) > 0.01:
            raise ValueError(
                f"relevance_weight + novelty_weight + impact_weight must equal 1.0 (got {total_openreview})"
            )


# Default configuration
DEFAULT_LLM_CONFIG = LLMConfig(
    model=LLMModel.GPT4O_MINI,
    temperature=0.0,
    max_tokens=1000,
)

DEFAULT_SCORING_WEIGHTS = ScoringWeights()

