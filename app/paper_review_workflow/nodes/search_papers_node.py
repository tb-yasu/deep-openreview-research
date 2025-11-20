"""Node for searching papers using OpenReview API."""

import json
from typing import Any

from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    Paper,
)
from app.paper_review_workflow.tools import search_papers


class SearchPapersNode:
    """Node for searching papers.
    
    Uses the OpenReview API to search for papers from specified conference/year
    and saves them to state.
    """
    
    def __init__(self) -> None:
        """Initialize SearchPapersNode."""
        self.tool = search_papers
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute paper search.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Dictionary of updated state
        """
        accepted_status = "Accepted only" if state.accepted_only else "All papers (accepted and rejected)"
        logger.info(
            f"Searching papers from {state.venue} {state.year} "
            f"(max: {state.max_papers}, keywords: {state.keywords}, {accepted_status})"
        )
        
        try:
            # Call tool to search papers
            result = self.tool.invoke({
                "venue": state.venue,
                "year": state.year,
                "keywords": state.keywords,
                "max_results": state.max_papers,
                "accepted_only": state.accepted_only,
            })
            
            # Parse result
            papers_data = json.loads(result)
            
            # Error check
            if isinstance(papers_data, dict) and "error" in papers_data:
                error_msg = f"Error searching papers: {papers_data['error']}"
                logger.error(error_msg)
                return {
                    "papers": [],
                    "error_messages": [error_msg],
                }
            
            # Convert to list of Paper objects
            papers: list[Paper] = []
            for paper_data in papers_data:
                try:
                    paper = Paper(**paper_data)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper data: {e}")
                    continue
            
            logger.info(f"Successfully found {len(papers)} papers")
            
            return {
                "papers": papers,
            }
            
        except Exception as e:
            error_msg = f"Unexpected error in SearchPapersNode: {e!s}"
            logger.error(error_msg)
            return {
                "papers": [],
                "error_messages": [error_msg],
            }

