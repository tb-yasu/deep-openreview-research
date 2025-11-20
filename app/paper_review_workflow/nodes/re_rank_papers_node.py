"""Node for re-ranking papers based on LLM evaluation scores."""

from typing import Any

from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
)
from app.paper_review_workflow.utils import convert_papers_to_dict_list
from app.paper_review_workflow.constants import DEFAULT_TOP_N_PAPERS


class ReRankPapersNode:
    """Node for re-ranking papers based on LLM evaluation scores."""
    
    def __init__(self, top_n: int = DEFAULT_TOP_N_PAPERS) -> None:
        """Initialize ReRankPapersNode.
        
        Args:
        ----
            top_n: Select top N papers (default: 20)
        """
        self.top_n = top_n
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Re-rank papers by LLM evaluation scores.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Dictionary of updated state
        """
        logger.info(f"Re-ranking {len(state.llm_evaluated_papers)} papers based on LLM scores...")
        logger.info(f"Top N papers to select: {self.top_n}")
        
        # Sort by overall_score (descending) - unified LLM evaluation system uses overall_score
        re_ranked_papers = sorted(
            state.llm_evaluated_papers,
            key=lambda p: p.overall_score if p.overall_score is not None else 0.0,
            reverse=True
        )
        
        # Assign rank numbers and convert to dict
        top_papers = convert_papers_to_dict_list(
            re_ranked_papers,
            max_count=self.top_n,
            include_llm_scores=True,
        )
        
        # Statistics
        if re_ranked_papers:
            scores = [p.overall_score for p in re_ranked_papers if p.overall_score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                logger.info(f"Average overall score: {avg_score:.3f}")
                top_score = re_ranked_papers[0].overall_score
                bottom_score = re_ranked_papers[-1].overall_score
                if top_score is not None:
                    logger.info(f"Top paper: {re_ranked_papers[0].title[:50]}... (Score: {top_score:.3f})")
                if bottom_score is not None:
                    logger.info(f"Bottom paper: {re_ranked_papers[-1].title[:50]}... (Score: {bottom_score:.3f})")
            else:
                logger.warning("No valid overall_score found in papers")
        
        logger.success(f"Re-ranked papers: {len(re_ranked_papers)} total, top {len(top_papers)} selected")
        
        return {
            "re_ranked_papers": re_ranked_papers,
            "top_papers": top_papers,
        }

