"""Utility functions for converting paper objects to dictionaries."""

from typing import Any

from app.paper_review_workflow.models.state import EvaluatedPaper


def convert_paper_to_dict(
    paper: EvaluatedPaper,
    rank: int | None = None,
    include_llm_scores: bool = False,
) -> dict[str, Any]:
    """Convert paper object to dictionary format.
    
    Args:
    ----
        paper: Evaluated paper object
        rank: Rank number (optional)
        include_llm_scores: Whether to include LLM evaluation scores
        
    Returns:
    -------
        Dictionary of paper information
    """
    paper_dict: dict[str, Any] = {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "keywords": paper.keywords,
        "overall_score": paper.overall_score,
        "relevance_score": paper.relevance_score,
        "novelty_score": paper.novelty_score,
        "impact_score": paper.impact_score,
        "practicality_score": paper.practicality_score,  # Unified LLM evaluation
        "rating_avg": paper.rating_avg,
        "reviews": paper.reviews,
        "decision": paper.decision,
        "forum_url": paper.forum_url,
        "pdf_url": paper.pdf_url,
        "evaluation_rationale": paper.evaluation_rationale,
        # New fields from unified LLM evaluation
        "review_summary": paper.review_summary,
        "field_insights": paper.field_insights,
        "ai_rationale": paper.ai_rationale,
        # OpenReview detailed information
        "meta_review": paper.meta_review,
        "decision_comment": paper.decision_comment,
        "author_remarks": paper.author_remarks,
    }
    
    # Add rank if provided
    if rank is not None:
        paper_dict["rank"] = rank
    
    # Include LLM scores if requested
    if include_llm_scores:
        paper_dict.update({
            "llm_relevance_score": paper.llm_relevance_score,
            "llm_novelty_score": paper.llm_novelty_score,
            "llm_practical_score": paper.llm_practical_score,
            "final_score": paper.final_score,
            "llm_rationale": paper.llm_rationale,
        })
    
    return paper_dict


def convert_papers_to_dict_list(
    papers: list[EvaluatedPaper],
    max_count: int | None = None,
    include_llm_scores: bool = False,
) -> list[dict[str, Any]]:
    """Convert paper list to list of dictionaries.
    
    Args:
    ----
        papers: List of evaluated papers
        max_count: Maximum number to convert (converts all if omitted)
        include_llm_scores: Whether to include LLM evaluation scores
        
    Returns:
    -------
        List of dictionaries of paper information
    """
    papers_to_convert = papers[:max_count] if max_count else papers
    
    return [
        convert_paper_to_dict(
            paper=paper,
            rank=i + 1,
            include_llm_scores=include_llm_scores,
        )
        for i, paper in enumerate(papers_to_convert)
    ]

