"""Node for fetching reviews on-demand for ranked papers."""

from typing import Any

from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
)


class FetchReviewsNode:
    """Node for fetching reviews on-demand.
    
    This node fetches review data for ranked papers that don't have reviews yet.
    It uses the review cache system to avoid repeated API calls.
    """
    
    def __init__(self) -> None:
        """Initialize FetchReviewsNode."""
        pass
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Fetch reviews for ranked papers.
        
        Args:
        ----
            state: Current state with ranked_papers
            
        Returns:
        -------
            Updated state dictionary with reviews merged into ranked_papers
        """
        if not state.ranked_papers:
            logger.warning("No ranked papers to fetch reviews for")
            return {}
        
        # Check if papers already have review data
        papers_without_reviews = [
            p for p in state.ranked_papers 
            if not p.reviews or len(p.reviews) == 0
        ]
        
        if not papers_without_reviews:
            logger.info("✓ All ranked papers already have review data")
            return {}
        
        logger.info(f"📚 {len(papers_without_reviews)}/{len(state.ranked_papers)} papers need review data")
        
        try:
            # Import here to avoid circular imports
            from review_cache import fetch_reviews_on_demand, merge_reviews_into_papers
            
            # Get paper IDs that need reviews
            paper_ids = [p.id for p in papers_without_reviews]
            
            # Fetch reviews on-demand (with caching)
            reviews = fetch_reviews_on_demand(
                paper_ids=paper_ids,
                venue=state.venue,
                year=state.year,
            )
            
            # Merge reviews into papers
            # Convert EvaluatedPaper objects to dicts for merging
            papers_as_dicts = [p.model_dump() for p in state.ranked_papers]
            merged_papers = merge_reviews_into_papers(papers_as_dicts, reviews)
            
            # Convert back to EvaluatedPaper objects
            updated_papers = []
            for paper_dict in merged_papers:
                try:
                    updated_paper = EvaluatedPaper(**paper_dict)
                    updated_papers.append(updated_paper)
                except Exception as e:
                    logger.warning(f"Failed to convert paper {paper_dict.get('id')}: {e}")
                    # Keep original paper
                    original = next((p for p in state.ranked_papers if p.id == paper_dict.get('id')), None)
                    if original:
                        updated_papers.append(original)
            
            # Count papers with reviews after merge
            papers_with_reviews = sum(1 for p in updated_papers if p.reviews and len(p.reviews) > 0)
            logger.success(f"✓ Reviews merged: {papers_with_reviews}/{len(updated_papers)} papers have review data")
            
            return {
                "ranked_papers": updated_papers,
            }
            
        except ImportError as e:
            logger.warning(f"⚠ review_cache module not available: {e}")
            logger.warning("⚠ Continuing without fetching reviews")
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to fetch reviews: {e}")
            logger.warning("⚠ Continuing without fetching reviews")
            return {}

