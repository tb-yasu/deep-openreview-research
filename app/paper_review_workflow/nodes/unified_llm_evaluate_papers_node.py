"""Unified LLM evaluation node - completes all evaluations in a single call."""

import asyncio
import json
import re
from typing import Any

from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
)
from app.paper_review_workflow.config import (
    LLMConfig,
    ScoringWeights,
    DEFAULT_SCORING_WEIGHTS,
)
from app.paper_review_workflow.constants import (
    MIN_SCORE,
    MAX_SCORE,
    MAX_AUTHORS_DISPLAY,
    MAX_KEYWORDS_DISPLAY,
)
from app.paper_review_workflow.llm_factory import create_chat_openai


class UnifiedLLMEvaluatePapersNode:
    """Unified LLM evaluation node - evaluates using title, abstract, and all review fields in one call."""
    
    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        scoring_weights: ScoringWeights | None = None,
    ) -> None:
        """Initialize UnifiedLLMEvaluatePapersNode.
        
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
            return create_chat_openai(
                model=model_name,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only OpenAI GPT models are supported.")
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute unified LLM evaluation (parallel version).
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Updated state dictionary
        """
        logger.info(f"🤖 Unified LLM evaluation for {len(state.ranked_papers)} papers using {self.llm_config.model.value}...")
        logger.info(f"⚡ Parallel execution with max 10 concurrent requests")
        logger.info(f"📊 Retrieving all scores + review summary + field insights in a single call")
        
        # Execute in parallel using asyncio event loop
        evaluated_papers = asyncio.run(
            self._evaluate_papers_parallel(
                state.ranked_papers,
                state.evaluation_criteria,
                max_concurrent=10,
            )
        )
        
        logger.success(f"✅ Successfully evaluated {len(evaluated_papers)} papers with unified LLM (parallel)")
        
        return {
            "llm_evaluated_papers": evaluated_papers,
        }
    
    async def _evaluate_papers_parallel(
        self,
        papers: list[EvaluatedPaper],
        criteria,
        max_concurrent: int = 10,
    ) -> list[EvaluatedPaper]:
        """Evaluate multiple papers in parallel (with rate limiting).
        
        Args:
        ----
            papers: List of papers to evaluate
            criteria: Evaluation criteria
            max_concurrent: Maximum concurrent requests (for API rate limiting)
            
        Returns:
        -------
            List of evaluated papers
        """
        # Limit concurrent execution with Semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(paper, index, total):
            async with semaphore:
                return await self._evaluate_single_paper_async(paper, criteria, index, total)
        
        # Execute all papers in parallel
        tasks = [
            evaluate_with_semaphore(paper, i + 1, len(papers))
            for i, paper in enumerate(papers)
        ]
        
        # Execute all tasks (wait for results)
        evaluated_papers = await asyncio.gather(*tasks, return_exceptions=False)
        
        return evaluated_papers
    
    async def _evaluate_single_paper_async(
        self,
        paper: EvaluatedPaper,
        criteria,
        index: int,
        total: int,
    ) -> EvaluatedPaper:
        """Evaluate a single paper asynchronously.
        
        Args:
        ----
            paper: Paper to evaluate
            criteria: Evaluation criteria
            index: Paper number (for logging)
            total: Total number of papers (for logging)
            
        Returns:
        -------
            Evaluated paper
        """
        try:
            logger.info(f"  [{index}/{total}] Evaluating: {paper.title[:50]}...")
            
            # Create unified prompt
            prompt = self._create_unified_evaluation_prompt(paper, criteria)
            
            # Request evaluation from LLM asynchronously
            response = await self.llm.ainvoke(prompt)
            response_text = response.content
            
            # Parse response
            evaluation = self._parse_llm_response(response_text)
            
            # Update paper object
            updated_paper = paper.model_copy(deep=True)
            updated_paper.relevance_score = evaluation['relevance']
            updated_paper.novelty_score = evaluation['novelty']
            updated_paper.impact_score = evaluation['impact']
            updated_paper.practicality_score = evaluation['practicality']
            updated_paper.review_summary = evaluation['review_summary']
            updated_paper.field_insights = evaluation['field_insights']
            updated_paper.ai_rationale = evaluation['rationale']
            
            # Calculate overall_score (weighted average of 4 scores)
            updated_paper.overall_score = (
                evaluation['relevance'] * 0.4 +
                evaluation['novelty'] * 0.25 +
                evaluation['impact'] * 0.25 +
                evaluation['practicality'] * 0.10
            )
            
            logger.debug(
                f"    ✓ [{index}/{total}] Scores: R={evaluation['relevance']:.2f} "
                f"N={evaluation['novelty']:.2f} "
                f"I={evaluation['impact']:.2f} "
                f"P={evaluation['practicality']:.2f} "
                f"Overall={updated_paper.overall_score:.2f}"
            )
            
            return updated_paper
            
        except Exception as e:
            logger.warning(f"  ⚠ Failed to evaluate paper {paper.id}: {e}")
            # Set default values on evaluation failure
            updated_paper = paper.model_copy(deep=True)
            updated_paper.relevance_score = 0.5
            updated_paper.novelty_score = 0.5
            updated_paper.impact_score = 0.5
            updated_paper.practicality_score = 0.5
            updated_paper.overall_score = 0.5
            updated_paper.review_summary = "Evaluation failed"
            updated_paper.field_insights = "N/A"
            updated_paper.ai_rationale = f"LLM evaluation error: {str(e)[:100]}"
            return updated_paper
    
    def _create_unified_evaluation_prompt(self, paper: EvaluatedPaper, criteria) -> str:
        """Create unified evaluation prompt - completes everything in one call."""
        
        # User's research interests
        research_interests_str = ", ".join(criteria.research_interests)
        user_interests = criteria.research_description or f"Keywords: {research_interests_str}"
        
        # Format review data
        reviews_formatted = self._format_dynamic_reviews(paper.reviews)
        
        prompt = f"""
You are an expert in evaluating machine learning research papers. Please comprehensively evaluate the following paper.

# 📄 Paper Information

**Title**: {paper.title}

**Authors**: {', '.join(paper.authors[:MAX_AUTHORS_DISPLAY])}{'...' if len(paper.authors) > MAX_AUTHORS_DISPLAY else ''}

**Keywords**: {', '.join(paper.keywords[:MAX_KEYWORDS_DISPLAY])}

**Abstract**:
{paper.abstract[:1500]}{'...' if len(paper.abstract) > 1500 else ''}

**Acceptance Decision**: {paper.decision or 'N/A'}

**Decision Comment** (Program Chairs):
{(paper.decision_comment[:500] + '...') if paper.decision_comment and len(paper.decision_comment) > 500 else (paper.decision_comment or 'N/A')}

# 📊 OpenReview Review Data

{reviews_formatted}

# 🎯 User's Research Interests

{user_interests}

# 📝 Evaluation Task

Please evaluate the following **4 scores** in the range 0.0-1.0:

## 1. Relevance
Evaluate the relevance to user's research interests.
- Judge from paper keywords, title, and abstract
- Reference "relevance" or "significance" fields in reviews if available

## 2. Novelty
Evaluate the originality and novelty of the research.
- Prioritize **"originality"** or **"novelty"** fields in reviews if available
- Reference descriptions of novelty in **"strengths_and_weaknesses"**
- Also reference **"claims_and_evidence"** or **"contribution"**
- Infer from abstract if unavailable

## 3. Impact
Evaluate academic and practical impact.
- Prioritize **"significance"** or **"contribution"** fields in reviews if available
- Also emphasize **"rating"** or **"overall_recommendation"**
- Consider acceptance decision (Accept/Reject)
- Reference quality of **"experimental_designs_or_analyses"**

## 4. Practicality
Evaluate practical applicability.
- Ease of implementation, reproducibility, potential for industrial application
- Also reference **"methods_and_evaluation_criteria"** or **"code_of_conduct"** fields
- Reference **"questions_for_authors"** in reviews

## 5. Review Summary
Integrate all reviews and summarize in 2-3 sentences:
- Main evaluation points from reviewers (strengths/weaknesses)
- Average evaluation tendency
- Program Chairs' decision reasoning (if available)

## 6. Field Insights
Explain in 1-2 sentences which review fields were primarily used:
Example: "Primarily referenced ICML's overall_recommendation field (average 3.0) and summary"
Example: "Primarily referenced NeurIPS's rating field (average 5.5) and strengths_and_weaknesses"

# Output Format

Please output ONLY the following JSON format (no explanation needed):

{{
  "relevance": 0.85,
  "novelty": 0.72,
  "impact": 0.68,
  "practicality": 0.80,
  "review_summary": "Reviewers highly praised the theoretical robustness of the method, while pointing out limitations in experiments. Program Chairs recommended acceptance based on balance of novelty and experimental quality.",
  "field_insights": "Primarily referenced ICML's overall_recommendation (average 2.75), theoretical_claims, and experimental_designs_or_analyses fields.",
  "rationale": "This paper specifically focuses on graph generation and is directly related to user's interests. New method with substantial experiments, but validation on large-scale datasets is limited."
}}
"""
        return prompt
    
    def _format_dynamic_reviews(self, reviews: list[dict]) -> str:
        """Format reviews with dynamic fields in a readable manner."""
        if not reviews:
            return "**No review data** (accepted but reviews are private, or retrieval error)"
        
        formatted_lines = []
        
        for i, review in enumerate(reviews, 1):
            formatted_lines.append(f"## Review {i}")
            formatted_lines.append("")
            
            # Display priority fields first
            priority_fields = {
                'rating': 'Score',
                'overall_recommendation': 'Score', 
                'confidence': 'Confidence',
                'summary': 'Summary',
            }
            
            for field, label in priority_fields.items():
                if field in review:
                    value = review[field]
                    # Truncate if too long
                    display = value[:300] + "..." if len(value) > 300 else value
                    formatted_lines.append(f"**{label}**: {display}")
            
            formatted_lines.append("")
            
            # Other fields (alphabetical order, max 10)
            other_fields = {k: v for k, v in review.items() 
                           if k not in priority_fields}
            
            if other_fields:
                formatted_lines.append("**Other Evaluation Items**:")
                for j, (field, value) in enumerate(sorted(other_fields.items()), 1):
                    if j > 10:  # Truncate if too long
                        formatted_lines.append(f"  ...{len(other_fields) - 10} more items")
                        break
                    # Make field name readable
                    field_display = field.replace('_', ' ').title()
                    # Truncate value
                    display = value[:150] + "..." if len(value) > 150 else value
                    formatted_lines.append(f"  • **{field_display}**: {display}")
            
            formatted_lines.append("")
            formatted_lines.append("---")
            formatted_lines.append("")
        
        return "\n".join(formatted_lines)
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response and extract evaluation results."""
        try:
            # Extract JSON block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON block, find {} from entire text
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Parse entire text as JSON
                    json_str = response.strip()
            
            # Parse JSON
            evaluation = json.loads(json_str)
            
            # Clip scores to 0-1 range
            return {
                'relevance': max(MIN_SCORE, min(MAX_SCORE, float(evaluation.get('relevance', 0.5)))),
                'novelty': max(MIN_SCORE, min(MAX_SCORE, float(evaluation.get('novelty', 0.5)))),
                'impact': max(MIN_SCORE, min(MAX_SCORE, float(evaluation.get('impact', 0.5)))),
                'practicality': max(MIN_SCORE, min(MAX_SCORE, float(evaluation.get('practicality', 0.5)))),
                'review_summary': str(evaluation.get('review_summary', 'No review summary'))[:500],
                'field_insights': str(evaluation.get('field_insights', 'No field information'))[:300],
                'rationale': str(evaluation.get('rationale', 'No evaluation rationale'))[:500],
            }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response: {response[:300]}...")
            # Return default values on parse failure
            return {
                'relevance': 0.5,
                'novelty': 0.5,
                'impact': 0.5,
                'practicality': 0.5,
                'review_summary': 'Failed to parse LLM evaluation',
                'field_insights': 'Parse error',
                'rationale': f'Parse error: {str(e)[:100]}',
            }
