"""Node for ranking evaluated papers."""

import asyncio
import re
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
    EvaluationCriteria,
)
from app.paper_review_workflow.utils import convert_papers_to_dict_list
from app.paper_review_workflow.constants import (
    MAX_DISPLAY_PAPERS,
    PRELIMINARY_LLM_MAX_TOKENS,
    ABSTRACT_SHORT_LENGTH,
    MAX_KEYWORDS_DISPLAY,
)
from app.paper_review_workflow.config import LLMConfig


class RankPapersNode:
    """Node for ranking evaluated papers by score."""
    
    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        """Initialize RankPapersNode.
        
        Args:
        ----
            llm_config: LLM configuration (includes concurrency settings)
        """
        self.llm = None  # Initialize when needed (cost reduction)
        self.llm_config = llm_config
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute paper ranking.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Dictionary of updated state
        """
        logger.info(f"Ranking {len(state.evaluated_papers)} evaluated papers...")
        
        # Filter based on evaluation criteria
        criteria = state.evaluation_criteria
        filtered_papers = [
            paper for paper in state.evaluated_papers
            if self._meets_criteria(paper, criteria)
        ]
        
        logger.info(
            f"After filtering: {len(filtered_papers)}/{len(state.evaluated_papers)} papers "
            f"meet the criteria"
        )
        
        # Sort by overall score (descending)
        ranked_papers = sorted(
            filtered_papers,
            key=lambda p: p.overall_score or 0.0,
            reverse=True,
        )
        
        # Preliminary LLM filter (if enabled)
        if criteria.enable_preliminary_llm_filter and len(ranked_papers) > 0:
            logger.info("🔍 Preliminary LLM filter enabled - evaluating top candidates...")
            ranked_papers = self._apply_preliminary_llm_filter(
                ranked_papers, 
                criteria
            )
        
        # Select top k papers if top_k is specified
        if criteria.top_k_papers is not None:
            selected_papers = ranked_papers[:criteria.top_k_papers]
            logger.info(
                f"Selected top {criteria.top_k_papers} papers from {len(ranked_papers)} ranked papers "
                f"(actual: {len(selected_papers)})"
            )
        else:
            selected_papers = ranked_papers
            logger.info(f"All {len(ranked_papers)} papers selected (no top_k limit)")
        
        # Convert top papers to dictionary format (for display)
        top_papers = convert_papers_to_dict_list(
            selected_papers,
            max_count=MAX_DISPLAY_PAPERS,
            include_llm_scores=False,
        )
        
        if top_papers:
            logger.success(f"Top paper: {top_papers[0]['title'][:50]} (Score: {top_papers[0]['overall_score']:.3f})")
        
        return {
            "ranked_papers": selected_papers,  # Paper list to pass to LLM evaluation
            "top_papers": top_papers,
        }
    
    def _meets_criteria(self, paper: EvaluatedPaper, criteria: EvaluationCriteria) -> bool:
        """Check if paper meets evaluation criteria.
        
        Args:
        ----
            paper: Evaluated paper
            criteria: Evaluation criteria
            
        Returns:
        -------
            True if criteria are met
        """
        # Check minimum relevance score
        if paper.relevance_score is not None:
            if paper.relevance_score < criteria.min_relevance_score:
                return False
        
        # Check minimum review score
        if criteria.min_rating is not None and paper.rating_avg is not None:
            if paper.rating_avg < criteria.min_rating:
                return False
        
        return True
    
    def _apply_preliminary_llm_filter(
        self, 
        ranked_papers: list[EvaluatedPaper], 
        criteria: EvaluationCriteria,
    ) -> list[EvaluatedPaper]:
        """Recalculate relevance_score with preliminary LLM evaluation and re-sort (parallel version).
        
        Args:
        ----
            ranked_papers: Sorted paper list
            criteria: Evaluation criteria
            
        Returns:
        -------
            Paper list with updated relevance_score and re-sorted
        """
        # Determine number of papers to evaluate
        filter_count = min(
            criteria.preliminary_llm_filter_count,
            len(ranked_papers)
        )
        
        # Determine concurrency (use llm_config if available, otherwise default to 10)
        max_concurrent = self.llm_config.max_concurrent if self.llm_config else 10
        
        logger.info(f"Evaluating top {filter_count} papers with LLM for better relevance scoring...")
        logger.info(f"⚡ Parallel execution with max {max_concurrent} concurrent requests")
        
        # Initialize LLM
        if self.llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=PRELIMINARY_LLM_MAX_TOKENS,
            )
        
        # Evaluate top N papers in parallel with LLM
        target_papers = ranked_papers[:filter_count]
        updated_papers = asyncio.run(
            self._evaluate_relevance_parallel(target_papers, criteria, max_concurrent=max_concurrent)
        )
        
        # Add remaining papers (not evaluated by LLM)
        remaining_papers = ranked_papers[filter_count:]
        all_papers = updated_papers + remaining_papers
        
        # Re-sort by relevance_score (reflected in overall_score, so sort by overall_score)
        re_ranked_papers = sorted(
            all_papers,
            key=lambda p: p.overall_score or 0.0,
            reverse=True,
        )
        
        # Count successes (papers not with default score)
        success_count = sum(1 for p in updated_papers if p.relevance_score != (ranked_papers[0].relevance_score or 0.0))
        
        logger.success(
            f"✓ Preliminary LLM filter completed: {success_count}/{filter_count} papers re-scored (parallel)"
        )
        
        return re_ranked_papers
    
    async def _evaluate_relevance_parallel(
        self,
        papers: list[EvaluatedPaper],
        criteria: EvaluationCriteria,
        max_concurrent: int = 10,
    ) -> list[EvaluatedPaper]:
        """Evaluate relevance of multiple papers in parallel.
        
        Args:
        ----
            papers: List of papers to evaluate
            criteria: Evaluation criteria
            max_concurrent: Maximum concurrent executions
            
        Returns:
        -------
            Updated paper list
        """
        # Limit concurrency with Semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(paper, index, total):
            async with semaphore:
                return await self._evaluate_single_relevance_async(paper, criteria, index, total)
        
        # Execute all papers in parallel
        tasks = [
            evaluate_with_semaphore(paper, i + 1, len(papers))
            for i, paper in enumerate(papers)
        ]
        
        # Execute all tasks (continue even if errors occur)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return only successfully completed papers (exclude Exceptions)
        updated_papers = [
            result for result in results
            if not isinstance(result, Exception)
        ]
        
        # Log number of papers that failed
        error_count = len(results) - len(updated_papers)
        if error_count > 0:
            logger.warning(f"⚠ {error_count}/{len(results)} papers failed during relevance evaluation")
        
        return updated_papers
    
    async def _evaluate_single_relevance_async(
        self,
        paper: EvaluatedPaper,
        criteria: EvaluationCriteria,
        index: int,
        total: int,
    ) -> EvaluatedPaper:
        """Evaluate relevance of a single paper asynchronously.
        
        Args:
        ----
            paper: Paper to evaluate
            criteria: Evaluation criteria
            index: Paper number (for logging)
            total: Total number of papers (for logging)
            
        Returns:
        -------
            Updated paper
        """
        try:
            # Evaluate relevance with LLM (async)
            llm_relevance = await self._evaluate_relevance_with_llm_async(paper, criteria)
            
            # Update relevance_score
            updated_paper = paper.model_copy(deep=True)
            old_score = paper.relevance_score or 0.0
            updated_paper.relevance_score = llm_relevance
            
            # Also update overall_score (considering relevance_weight)
            score_diff = llm_relevance - old_score
            updated_paper.overall_score = (paper.overall_score or 0.0) + score_diff * 0.4  # relevance_weight=0.4
            
            if index % 50 == 0:
                logger.info(f"  Progress: {index}/{total} papers evaluated")
            
            return updated_paper
            
        except Exception as e:
            logger.warning(f"Failed to LLM evaluate paper {paper.id}: {e}")
            # Keep original score on failure
            return paper
    
    async def _evaluate_relevance_with_llm_async(
        self,
        paper: EvaluatedPaper,
        criteria: EvaluationCriteria,
    ) -> float:
        """Quickly evaluate paper relevance with LLM (async version).
        
        Args:
        ----
            paper: Paper to evaluate
            criteria: Evaluation criteria
            
        Returns:
        -------
            Relevance score (0.0-1.0)
        """
        # Shorten abstract
        abstract_short = (
            paper.abstract[:ABSTRACT_SHORT_LENGTH] + 
            ("..." if len(paper.abstract) > ABSTRACT_SHORT_LENGTH else "")
        )
        
        # Convert user interests to string
        interests_str = ", ".join(criteria.research_interests)
        user_description = criteria.research_description or f"Keywords: {interests_str}"
        
        # Create prompt
        prompt = f"""
Evaluate how relevant the following paper is to the user's research interests on a scale of 0.0-1.0.

# Paper Information

**Title**: {paper.title}

**Keywords**: {', '.join(paper.keywords[:MAX_KEYWORDS_DISPLAY])}

**Abstract**:
{abstract_short}

# User's Research Interests

{user_description}

# Output Format

Output only the score in the range of 0.0-1.0 (e.g., 0.85)
"""
        
        # Query LLM asynchronously
        response = await self.llm.ainvoke(prompt)
        response_text = response.content.strip()
        
        # Extract score
        score_match = re.search(r'(0\.\d+|1\.0|0|1)', response_text)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        else:
            logger.warning(f"Failed to parse relevance score from: {response_text[:100]}")
            return paper.relevance_score or 0.5
    
    def _evaluate_relevance_with_llm(
        self, 
        paper: EvaluatedPaper, 
        criteria: EvaluationCriteria,
    ) -> float:
        """Quickly evaluate paper relevance with LLM.
        
        Args:
        ----
            paper: Paper to evaluate
            criteria: Evaluation criteria
            
        Returns:
        -------
            Relevance score (0.0-1.0)
        """
        # Shorten abstract
        abstract_short = (
            paper.abstract[:ABSTRACT_SHORT_LENGTH] + 
            ("..." if len(paper.abstract) > ABSTRACT_SHORT_LENGTH else "")
        )
        keywords_str = ", ".join(paper.keywords[:MAX_KEYWORDS_DISPLAY])
        
        # Fallback to research_interests if research_description is not available
        research_interests_str = ", ".join(criteria.research_interests)
        user_interests = criteria.research_description or f"Keywords: {research_interests_str}"
        
        prompt = f"""Rate the relevance of this paper to the user's research interests.

User's Research Interests:
{user_interests}

Paper:
Title: {paper.title}
Keywords: {keywords_str}
Abstract: {abstract_short}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly addresses the research interests
- 0.7-0.9: Very relevant, closely related
- 0.4-0.6: Moderately relevant, some overlap
- 0.1-0.3: Slightly relevant, tangential connection
- 0.0: Not relevant

Return ONLY a single number between 0.0 and 1.0 (e.g., "0.85"). No other text.
"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Extract numeric value
            # Handle formats like "0.85" or "The relevance is 0.85"
            match = re.search(r'(\d+\.?\d*)', response_text)
            if match:
                score = float(match.group(1))
                # Limit to 0-1 range
                score = max(0.0, min(1.0, score))
                return score
            else:
                logger.warning(f"Could not parse LLM response: {response_text[:50]}")
                return paper.relevance_score or 0.5
                
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return paper.relevance_score or 0.5

