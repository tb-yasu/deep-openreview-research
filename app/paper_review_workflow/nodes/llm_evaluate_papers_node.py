"""Node for evaluating papers using LLM."""

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
)
from app.paper_review_workflow.config import (
    LLMConfig,
    LLMModel,
    ScoringWeights,
    DEFAULT_SCORING_WEIGHTS,
)
from app.paper_review_workflow.constants import (
    MIN_SCORE,
    MAX_SCORE,
    MAX_AUTHORS_DISPLAY,
    MAX_KEYWORDS_DISPLAY,
    MAX_RATIONALE_LENGTH,
)


class LLMEvaluatePapersNode:
    """Node for deeply evaluating paper content using LLM."""
    
    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        scoring_weights: ScoringWeights | None = None,
    ) -> None:
        """Initialize LLMEvaluatePapersNode.
        
        Args:
        ----
            llm_config: LLM configuration (uses default if omitted)
            scoring_weights: Scoring weight configuration (uses default if omitted)
        """
        from app.paper_review_workflow.config import DEFAULT_LLM_CONFIG
        
        self.llm_config = llm_config or DEFAULT_LLM_CONFIG
        self.weights = scoring_weights or DEFAULT_SCORING_WEIGHTS
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create LLM instance."""
        model_name = self.llm_config.model.value
        
        if model_name.startswith("gpt"):
            return ChatOpenAI(
                model=model_name,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only OpenAI GPT models are supported.")
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute LLM evaluation.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Dictionary of updated state
        """
        logger.info(f"LLM evaluating {len(state.ranked_papers)} papers using {self.llm_config.model.value}...")
        
        llm_evaluated_papers: list[EvaluatedPaper] = []
        
        for i, paper in enumerate(state.ranked_papers, 1):
            try:
                logger.info(f"LLM evaluating paper {i}/{len(state.ranked_papers)}: {paper.title[:50]}...")
                
                # Create prompt
                prompt = self._create_evaluation_prompt(paper, state.evaluation_criteria)
                
                # Request evaluation from LLM
                response = self.llm.invoke(prompt)
                response_text = response.content
                
                # Parse scores
                scores = self._parse_llm_response(response_text)
                
                # Update paper object
                updated_paper = paper.model_copy(deep=True)
                updated_paper.llm_relevance_score = scores['relevance']
                updated_paper.llm_novelty_score = scores['novelty']
                updated_paper.llm_practical_score = scores['practical']
                updated_paper.llm_rationale = scores['rationale']
                
                # Calculate final score (integrated with configured weights)
                llm_average = (scores['relevance'] + scores['novelty'] + scores['practical']) / 3
                updated_paper.final_score = (
                    paper.overall_score * self.weights.openreview_weight +
                    llm_average * self.weights.llm_weight
                )
                
                llm_evaluated_papers.append(updated_paper)
                
                logger.debug(f"LLM scores - Relevance: {scores['relevance']:.3f}, "
                           f"Novelty: {scores['novelty']:.3f}, "
                           f"Practical: {scores['practical']:.3f}, "
                           f"Final: {updated_paper.final_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to LLM evaluate paper {paper.id}: {e}")
                # Keep original score on LLM evaluation failure
                updated_paper = paper.model_copy(deep=True)
                updated_paper.final_score = paper.overall_score
                llm_evaluated_papers.append(updated_paper)
                continue
        
        logger.success(f"Successfully LLM evaluated {len(llm_evaluated_papers)} papers")
        
        return {
            "llm_evaluated_papers": llm_evaluated_papers,
        }
    
    def _create_evaluation_prompt(self, paper: EvaluatedPaper, criteria) -> str:
        """Create evaluation prompt."""
        research_interests_str = ", ".join(criteria.research_interests)
        
        # Fallback to research_interests if research_description is not available
        user_interests = criteria.research_description or f"Keywords: {research_interests_str}"

        prompt = f"""
            Please evaluate the following paper.

            # Paper Information
Title: {paper.title}
Authors: {', '.join(paper.authors[:MAX_AUTHORS_DISPLAY])}{'...' if len(paper.authors) > MAX_AUTHORS_DISPLAY else ''}
Keywords: {', '.join(paper.keywords[:MAX_KEYWORDS_DISPLAY])}
Abstract:
{paper.abstract}
OpenReview Rating (Reference): {paper.rating_avg if paper.rating_avg else 'N/A'}/10

# Instructions
Please read and evaluate based on the title, abstract, keywords, and OpenReview rating.

# User's Research Interests
{user_interests}

# Evaluation Criteria (evaluate with real values from 0.0 to 1.0)
1. Relevance
2. Novelty
3. Practicality

# Output Format
Output only JSON in the following format:

{{
  "relevance": float,
  "novelty": float,
  "practical": float,
  "rationale": "Briefly explain the reason for each score in 2-3 sentences"
}}
"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response and extract scores."""
        try:
            # Extract JSON block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON block, parse entire response as JSON
                json_str = response.strip()
            
            # Parse JSON
            scores = json.loads(json_str)
            
            # Clip scores to 0-1 range
            return {
                'relevance': max(MIN_SCORE, min(MAX_SCORE, float(scores.get('relevance', 0.5)))),
                'novelty': max(MIN_SCORE, min(MAX_SCORE, float(scores.get('novelty', 0.5)))),
                'practical': max(MIN_SCORE, min(MAX_SCORE, float(scores.get('practical', 0.5)))),
                'rationale': str(scores.get('rationale', 'No evaluation rationale'))[:MAX_RATIONALE_LENGTH],
            }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response: {response[:200]}...")
            # Default values on parse failure
            return {
                'relevance': 0.5,
                'novelty': 0.5,
                'practical': 0.5,
                'rationale': 'Failed to parse LLM evaluation',
            }

