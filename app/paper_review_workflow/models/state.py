"""State management for Paper Review Agent."""

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Model representing basic paper information."""
    
    id: str = Field(title="Paper ID")
    title: str = Field(title="Paper Title")
    authors: list[str] = Field(default_factory=list, title="Author List")
    abstract: str = Field(default="", title="Abstract")
    keywords: list[str] = Field(default_factory=list, title="Keyword List")
    venue: str = Field(title="Conference Name")
    year: int = Field(title="Conference Year")
    pdf_url: str = Field(title="PDF URL")
    forum_url: str = Field(title="Forum URL")
    
    # Review data (included if already fetched by fetch_all_papers.py)
    reviews: list[dict[str, Any]] = Field(default_factory=list, title="Review List")
    rating_avg: float | None = Field(default=None, title="Average Rating")
    confidence_avg: float | None = Field(default=None, title="Average Confidence")
    decision: str | None = Field(default=None, title="Acceptance Decision")
    
    # Additional OpenReview information
    meta_review: str | None = Field(default=None, title="Meta Review (Area Chair Summary)")
    author_remarks: str | None = Field(default=None, title="Author's Final Comments")
    decision_comment: str | None = Field(default=None, title="Detailed Decision Comment")


class EvaluatedPaper(Paper):
    """Model representing evaluated paper information."""
    
    # Unified LLM evaluation scores (new system)
    relevance_score: float | None = Field(
        default=None,
        title="Relevance Score",
        description="Relevance to user's research interests (0.0-1.0)",
    )
    novelty_score: float | None = Field(
        default=None,
        title="Novelty Score",
        description="Novelty and originality of research (0.0-1.0)",
    )
    impact_score: float | None = Field(
        default=None,
        title="Impact Score",
        description="Potential impact of research (0.0-1.0)",
    )
    practicality_score: float | None = Field(
        default=None,
        title="Practicality Score",
        description="Practical applicability (0.0-1.0)",
    )
    overall_score: float | None = Field(
        default=None,
        title="Overall Score",
        description="Weighted average of 4 scores (0.0-1.0)",
    )
    
    # Additional information from unified LLM evaluation
    review_summary: str | None = Field(
        default=None,
        title="Review Summary",
        description="Integrated summary of OpenReview reviews",
    )
    field_insights: str | None = Field(
        default=None,
        title="Field Usage Explanation",
        description="Explanation of which review fields were primarily used",
    )
    ai_rationale: str | None = Field(
        default=None,
        title="AI Evaluation Rationale",
        description="Detailed reasoning for AI evaluation (2-3 sentences)",
    )
    
    # Backward compatibility with old system
    evaluation_rationale: str = Field(
        default="",
        title="Evaluation Rationale (Old)",
        description="Detailed reasoning for scores (kept for backward compatibility)",
    )
    citation_count: int | None = Field(
        default=None,
        title="Citation Count",
        description="Citation count from Semantic Scholar",
    )
    
    # Old LLM evaluation scores (kept for backward compatibility)
    llm_relevance_score: float | None = Field(
        default=None,
        title="LLM Relevance Score (Old)",
        description="LLM relevance score from old system (0.0-1.0)",
    )
    llm_novelty_score: float | None = Field(
        default=None,
        title="LLM Novelty Score (Old)",
        description="LLM novelty score from old system (0.0-1.0)",
    )
    llm_practical_score: float | None = Field(
        default=None,
        title="LLM Practicality Score (Old)",
        description="LLM practicality score from old system (0.0-1.0)",
    )
    llm_rationale: str | None = Field(
        default=None,
        title="LLM Evaluation Rationale (Old)",
        description="LLM evaluation rationale from old system",
    )
    final_score: float | None = Field(
        default=None,
        title="Final Score (Old)",
        description="Final score from old system (0.0-1.0)",
    )
    rank: int | None = Field(
        default=None,
        title="Rank",
        description="Paper rank (starting from 1)",
    )


class EvaluationCriteria(BaseModel):
    """Model representing paper evaluation criteria."""
    
    research_description: str | None = Field(
        default=None,
        title="Research Interest Description",
        description="Natural language description of user's research interests",
    )
    research_interests: list[str] = Field(
        default_factory=list,
        title="Research Interest Keywords",
        description="List of keywords for user's research interests/areas (extracted from research_description if omitted)",
    )
    min_relevance_score: float = Field(
        default=0.5,
        title="Minimum Relevance Score",
        description="Minimum relevance score to consider a paper valuable (0.0-1.0)",
    )
    min_rating: float | None = Field(
        default=None,
        title="Minimum Review Rating",
        description="Minimum OpenReview rating for papers (optional)",
    )
    min_citations: int | None = Field(
        default=None,
        title="Minimum Citation Count",
        description="Minimum citation count for papers (optional)",
    )
    focus_on_novelty: bool = Field(
        default=True,
        title="Focus on Novelty",
        description="Whether to emphasize novelty",
    )
    focus_on_impact: bool = Field(
        default=True,
        title="Focus on Impact",
        description="Whether to emphasize impact",
    )
    top_k_papers: int | None = Field(
        default=None,
        title="Top K Papers",
        description="Number of top papers to evaluate with LLM (filters by threshold only if None)",
    )
    enable_preliminary_llm_filter: bool = Field(
        default=False,
        title="Enable Preliminary LLM Filter",
        description="Whether to recalculate relevance_score with preliminary LLM evaluation before top-k selection",
    )
    preliminary_llm_filter_count: int = Field(
        default=500,
        title="Preliminary LLM Filter Count",
        description="Number of candidate papers to evaluate with preliminary LLM (evaluate top N to improve accuracy)",
    )


class PaperReviewAgentInputState(BaseModel):
    """Input state for PaperReviewAgent."""
    
    venue: str = Field(
        title="Conference Name",
        description="Conference name to search (e.g., 'NeurIPS', 'ICML', 'ICLR')",
    )
    year: int = Field(
        title="Conference Year",
        description="Conference year to search (e.g., 2024)",
    )
    keywords: str | None = Field(
        default=None,
        title="Search Keywords",
        description="Keywords to filter papers (optional)",
    )
    max_papers: int = Field(
        default=50,
        title="Maximum Papers",
        description="Maximum number of papers to search",
    )
    accepted_only: bool = Field(
        default=True,
        title="Accepted Only",
        description="Search only accepted papers (default: True)",
    )
    evaluation_criteria: EvaluationCriteria = Field(
        default_factory=EvaluationCriteria,
        title="Evaluation Criteria",
        description="Criteria for evaluating papers",
    )


class PaperReviewAgentPrivateState(BaseModel):
    """Internal state for PaperReviewAgent."""
    
    papers: list[Paper] = Field(
        default_factory=list,
        title="Searched Paper List",
    )
    evaluated_papers: Annotated[list[EvaluatedPaper], operator.add] = Field(
        default_factory=list,
        title="Evaluated Paper List",
    )
    ranked_papers: list[EvaluatedPaper] = Field(
        default_factory=list,
        title="Ranked Paper List",
    )
    llm_evaluated_papers: list[EvaluatedPaper] = Field(
        default_factory=list,
        title="LLM-Evaluated Paper List",
    )
    re_ranked_papers: list[EvaluatedPaper] = Field(
        default_factory=list,
        title="Paper List Re-ranked by LLM Scores",
    )
    synonyms: dict[str, list[str]] = Field(
        default_factory=dict,
        title="Keyword Synonym Dictionary",
        description="Mapping of each keyword to its synonyms (key: keyword, value: synonym list)",
    )
    error_messages: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        title="Error Message List",
    )


class PaperReviewAgentOutputState(BaseModel):
    """Output state for PaperReviewAgent."""
    
    paper_report: str | None = Field(
        default=None,
        title="Paper Review Report",
        description=(
            "Final report generated based on searched and evaluated papers. "
            "Includes overview, evaluation rationale, and recommendations for top valuable papers."
        ),
    )
    top_papers: list[dict[str, Any]] = Field(
        default_factory=list,
        title="Top Paper List",
        description="List of top papers with high evaluation scores",
    )


class PaperReviewAgentState(
    PaperReviewAgentInputState,
    PaperReviewAgentPrivateState,
    PaperReviewAgentOutputState,
):
    """Complete state for PaperReviewAgent."""
    pass

