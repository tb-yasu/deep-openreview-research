"""Node for evaluating papers based on OpenReview review data."""

import json
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
    Paper,
    EvaluationCriteria,
)
from app.paper_review_workflow.tools import fetch_paper_metadata
from app.paper_review_workflow.config import ScoringWeights, DEFAULT_SCORING_WEIGHTS
from app.paper_review_workflow.constants import (
    SYNONYMS_LLM_MAX_TOKENS,
    SYNONYMS_COUNT_MIN,
    SYNONYMS_COUNT_MAX,
    MIN_SCORE,
    MAX_SCORE,
    NEURIPS_RATING_SCALE,
    RELEVANCE_KEYWORD_WEIGHT,
    RELEVANCE_TEXT_WEIGHT,
    RELEVANCE_COVERAGE_WEIGHT,
    MAX_RATIONALE_LENGTH,
)


class EvaluatePapersNode:
    """Node for evaluating papers based on OpenReview review data."""
    
    def __init__(self, scoring_weights: ScoringWeights | None = None) -> None:
        """Initialize EvaluatePapersNode.
        
        Args:
        ----
            scoring_weights: Scoring weight configuration (uses default if omitted)
        """
        self.tool = fetch_paper_metadata
        self.weights = scoring_weights or DEFAULT_SCORING_WEIGHTS
        self._synonyms_cache: dict[str, list[str]] = {}  # Synonym cache
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute paper evaluation.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Dictionary of updated state
        """
        logger.info(f"Evaluating {len(state.papers)} papers based on review data...")
        
        # Generate synonyms first (used for evaluating all papers)
        research_interests = state.evaluation_criteria.research_interests
        if research_interests:
            synonyms = self._generate_synonyms(research_interests)
        else:
            synonyms = {}
        
        evaluated_papers: list[EvaluatedPaper] = []
        
        for i, paper in enumerate(state.papers, 1):
            try:
                logger.info(f"Evaluating paper {i}/{len(state.papers)}: {paper.title[:50]}...")
                
                # Get metadata (use existing review data if available)
                if paper.reviews and paper.rating_avg is not None:
                    # Use data loaded from all_papers.json (no API call needed)
                    logger.debug(f"Using cached review data for {paper.id}")
                    metadata = {
                        "reviews": paper.reviews,
                        "rating_avg": paper.rating_avg,
                        "confidence_avg": paper.confidence_avg,
                        "decision": paper.decision,
                    }
                else:
                    # Fetch from API
                    logger.debug(f"Fetching review data from API for {paper.id}")
                    result = self.tool.invoke({"paper_id": paper.id})
                    metadata = json.loads(result)
                    
                    # Error check
                    if isinstance(metadata, dict) and "error" in metadata:
                        logger.warning(f"Failed to fetch metadata for {paper.id}: {metadata['error']}")
                        # Evaluate without metadata
                        evaluated_paper = EvaluatedPaper(
                            **paper.model_dump(),
                            relevance_score=None,
                            novelty_score=None,
                            impact_score=None,
                            overall_score=0.0,
                            evaluation_rationale="Failed to fetch metadata",
                        )
                        evaluated_papers.append(evaluated_paper)
                        continue
                
                # Calculate scores (also pass paper object)
                scores = self._calculate_scores(paper, metadata, state.evaluation_criteria)
                
                # Generate evaluation rationale
                rationale = self._generate_rationale(metadata, scores)
                
                # Create EvaluatedPaper object
                evaluated_paper = EvaluatedPaper(
                    **paper.model_dump(),
                    relevance_score=scores["relevance"],
                    novelty_score=scores["novelty"],
                    impact_score=scores["impact"],
                    overall_score=scores["overall"],
                    evaluation_rationale=rationale,
                )
                
                evaluated_papers.append(evaluated_paper)
                logger.debug(f"Evaluated: {paper.title[:50]} - Score: {scores['overall']:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating paper {paper.id}: {e}")
                # Don't skip even if error occurs, add with score 0
                evaluated_paper = EvaluatedPaper(
                    **paper.model_dump(),
                    overall_score=0.0,
                    evaluation_rationale=f"Evaluation error: {e!s}",
                )
                evaluated_papers.append(evaluated_paper)
        
        logger.info(f"Successfully evaluated {len(evaluated_papers)} papers")
        
        return {
            "evaluated_papers": evaluated_papers,
            "synonyms": synonyms,
        }
    
    def _calculate_scores(
        self,
        paper: Paper,
        metadata: dict[str, Any],
        criteria: EvaluationCriteria,
    ) -> dict[str, float]:
        """Calculate various scores from review data.
        
        Args:
        ----
            paper: Paper object
            metadata: Paper metadata
            criteria: Evaluation criteria
            
        Returns:
        -------
            Dictionary of various scores
        """
        rating_avg = metadata.get("rating_avg")
        
        if rating_avg is None:
            # If no review data: keyword-based relevance only
            relevance = self._calculate_relevance_score(paper, criteria)
            return {
                "relevance": relevance,
                "novelty": 0.5,     # Neutral
                "impact": 0.5,      # Neutral
                "overall": relevance * 0.7 + 0.3,  # Emphasize relevance
            }
        
        # Normalize review score to 0-1 scale
        normalized_rating = rating_avg / NEURIPS_RATING_SCALE
        
        # 1. Relevance score: Matching with user's research interests (keyword-based only)
        relevance_score = self._calculate_relevance_score(paper, criteria)
        
        # 2. Novelty score: Estimated from review content (improved version)
        novelty_score = self._estimate_novelty_from_reviews(metadata, normalized_rating)
        
        # 3. Impact score: Calculated from acceptance decision and review score
        impact_score = self._calculate_impact_score(metadata, normalized_rating)
        
        # 4. Overall score: Integrated with configured weights (no duplication)
        overall_score = (
            relevance_score * self.weights.relevance_weight +
            novelty_score * self.weights.novelty_weight +
            impact_score * self.weights.impact_weight
        )
        
        return {
            "relevance": min(max(relevance_score, MIN_SCORE), MAX_SCORE),
            "novelty": min(max(novelty_score, MIN_SCORE), MAX_SCORE),
            "impact": min(max(impact_score, MIN_SCORE), MAX_SCORE),
            "overall": min(max(overall_score, MIN_SCORE), MAX_SCORE),
        }
    
    def _generate_synonyms(self, research_interests: list[str]) -> dict[str, list[str]]:
        """Generate synonyms using LLM for each keyword.
        
        Processing each keyword individually ensures that keywords and synonym dictionary keys
        match exactly.
        
        Args:
        ----
            research_interests: List of user's research interest keywords
            
        Returns:
        -------
            Synonym dictionary per keyword (key: original keyword, value: synonym list)
        """
        # Check cache
        cache_key = ",".join(sorted(research_interests))
        if cache_key in self._synonyms_cache:
            logger.debug("Using cached synonyms")
            return self._synonyms_cache[cache_key]
        
        logger.info(f"Generating synonyms for {len(research_interests)} research interests using LLM...")
        
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=SYNONYMS_LLM_MAX_TOKENS,
            )
            
            # Generate synonyms individually for each keyword
            synonyms = {}
            
            for keyword in research_interests:
                keyword_lower = keyword.lower().strip()
                
                prompt = f"""Generate {SYNONYMS_COUNT_MIN}-{SYNONYMS_COUNT_MAX} synonyms and related terms for this research topic:

Topic: "{keyword}"

**IMPORTANT**: For EVERY synonym you generate, include BOTH singular and plural forms.

Return ONLY a JSON array of synonyms (all lowercase):
["synonym1", "synonym2", "synonym3", ...]

Rules:
- **Output synonyms in ENGLISH only** (even if input topic is in another language)
- **For each concept, provide both singular and plural**
  Examples:
  * Topic "ai agents" → Include: ["ai agent", "ai agents", "intelligent agent", "intelligent agents", ...]
  * Topic "distributed system" → Include: ["distributed system", "distributed systems", "decentralized system", "decentralized systems", ...]
- Common abbreviations (e.g., "llm" for "large language model")
- Related terms and alternative phrasings
- Keep terms concise and technical

Remember: Always include both "X system" AND "X systems", "Y agent" AND "Y agents", etc.
"""
                
                try:
                    response = llm.invoke(prompt)
                    response_text = response.content.strip()
                    
                    # Parse JSON (remove code blocks)
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    syn_list = json.loads(response_text)
                    
                    # Process only if it's a list
                    if isinstance(syn_list, list):
                        # Lowercase and remove duplicates
                        synonyms[keyword_lower] = [s.lower().strip() for s in syn_list if s]
                        logger.debug(f"  ✓ '{keyword_lower}': {synonyms[keyword_lower][:3]}...")
                    else:
                        logger.warning(f"Invalid synonym format for '{keyword}': expected list, got {type(syn_list)}")
                        synonyms[keyword_lower] = []
                
                except Exception as e:
                    logger.warning(f"Failed to generate synonyms for '{keyword}': {e}")
                    # Set empty list on error (skip only that keyword)
                    synonyms[keyword_lower] = []
            
            # Save to cache
            self._synonyms_cache[cache_key] = synonyms
            
            # Summary log
            successful = sum(1 for syns in synonyms.values() if syns)
            logger.success(f"Generated synonyms for {successful}/{len(synonyms)} topics")
            
            return synonyms
            
        except Exception as e:
            logger.warning(f"Failed to generate synonyms: {e}. Using original keywords only.")
            # Return empty dictionary on error (use original keywords only)
            return {}
    
    def _calculate_relevance_score(
        self,
        paper: Paper,
        criteria: EvaluationCriteria,
    ) -> float:
        """Calculate relevance to user's research interests (group-based).
        
        Determines matches for each keyword group (original keyword + synonyms),
        counting multiple matches within the same group as one.
        Paper keyword matches are weighted higher than title/abstract matches.
        
        Args:
        ----
            paper: Paper object
            criteria: Evaluation criteria
            
        Returns:
        -------
            Relevance score (0.0-1.0)
        """
        research_interests = criteria.research_interests
        
        if not research_interests:
            # Neutral if no research interests specified
            return 0.5
        
        # Generate synonyms (LLM call only on first time, then cached)
        synonyms = self._generate_synonyms(research_interests)
        
        # Prepare paper data
        paper_keywords = set([kw.lower().strip() for kw in paper.keywords])
        paper_text = (paper.title + " " + paper.abstract).lower()
        
        # Determine matches for each keyword group
        matched_groups = 0
        matched_in_paper_keywords = 0
        matched_in_text_only = 0
        
        for interest in research_interests:
            interest_lower = interest.lower().strip()
            
            # Group keywords (original keyword + synonyms)
            group_keywords = {interest_lower}
            if interest_lower in synonyms:
                group_keywords.update([syn.lower().strip() for syn in synonyms[interest_lower]])
            
            # Check if this group matches paper keywords
            has_keyword_match = bool(group_keywords & paper_keywords)
            
            # Check if this group matches title/abstract
            has_text_match = any(kw in paper_text for kw in group_keywords)
            
            if has_keyword_match or has_text_match:
                matched_groups += 1
                
                if has_keyword_match:
                    matched_in_paper_keywords += 1
                elif has_text_match:
                    matched_in_text_only += 1
        
        # Score calculation (weights designed so total max is 1.0)
        num_groups = len(research_interests)
        
        # Prioritize paper keyword matches
        keyword_weight_per_group = RELEVANCE_KEYWORD_WEIGHT / num_groups if num_groups > 0 else 0
        
        # Text matches are weighted lower
        text_weight_per_group = RELEVANCE_TEXT_WEIGHT / num_groups if num_groups > 0 else 0
        
        # Coverage bonus
        coverage_weight = RELEVANCE_COVERAGE_WEIGHT
        
        keyword_match_score = matched_in_paper_keywords * keyword_weight_per_group
        text_match_score = matched_in_text_only * text_weight_per_group
        coverage_score = (matched_groups / num_groups) * coverage_weight
        
        total_score = keyword_match_score + text_match_score + coverage_score
        
        logger.debug(
            f"Relevance: {total_score:.3f} "
            f"(keyword:{matched_in_paper_keywords}/{num_groups}={keyword_match_score:.3f}, "
            f"text:{matched_in_text_only}/{num_groups}={text_match_score:.3f}, "
            f"coverage:{matched_groups}/{num_groups}={coverage_score:.3f})"
        )
        
        # Theoretical maximum is sum of configured weights
        # Keep min() as precaution, but normally won't exceed 1.0
        return min(MAX_SCORE, total_score)
    
    def _calculate_impact_score(
        self,
        metadata: dict[str, Any],
        normalized_rating: float,
    ) -> float:
        """Calculate research impact.
        
        Args:
        ----
            metadata: Paper metadata
            normalized_rating: Normalized review score
            
        Returns:
        -------
            Impact score (0.0-1.0)
        """
        # Impact of acceptance decision
        decision = metadata.get("decision", "").lower()
        decision_score = 0.5  # Default
        
        if "oral" in decision or "spotlight" in decision:
            decision_score = 1.0  # High rating
        elif "accept" in decision:
            decision_score = 0.7
        elif "reject" in decision:
            decision_score = 0.2
        
        # Reviewer confidence
        confidence_avg = metadata.get("confidence_avg")
        confidence_score = (confidence_avg / 5.0) if confidence_avg else 0.5
        
        # Impact score = Decision 50% + Review score 30% + Confidence 20%
        impact = (
            decision_score * 0.5 +
            normalized_rating * 0.3 +
            confidence_score * 0.2
        )
        
        return min(MAX_SCORE, max(MIN_SCORE, impact))
    
    def _estimate_novelty_from_reviews(
        self,
        metadata: dict[str, Any],
        normalized_rating: float,
    ) -> float:
        """Estimate novelty from review content (improved version).
        
        Args:
        ----
            metadata: Paper metadata
            normalized_rating: Normalized review score
            
        Returns:
        -------
            Novelty score (0.0-1.0)
        """
        reviews = metadata.get("reviews", [])
        if not reviews:
            return normalized_rating  # Use overall rating if no reviews
        
        # Keywords related to novelty (positive)
        positive_keywords = [
            "novel", "new approach", "innovative", "original", "first",
            "groundbreaking", "pioneering", "unique", "creative", "fresh"
        ]
        
        # Keywords indicating low novelty (negative)
        negative_keywords = [
            "not novel", "incremental", "limited novelty", "similar to",
            "existing work", "well-known", "standard approach"
        ]
        
        positive_score = 0
        negative_score = 0
        
        for review in reviews:
            strengths = review.get("strengths", "").lower()
            weaknesses = review.get("weaknesses", "").lower()
            summary = review.get("summary", "").lower()
            
            # Combine review text
            review_text = strengths + " " + weaknesses + " " + summary
            
            # Count positive mentions
            for keyword in positive_keywords:
                if keyword in review_text:
                    # Mentions in strengths are weighted 2x
                    if keyword in strengths:
                        positive_score += 2
                    else:
                        positive_score += 1
        
            # Count negative mentions
            for keyword in negative_keywords:
                if keyword in review_text:
                    # Mentions in weaknesses are weighted 2x
                    if keyword in weaknesses:
                        negative_score += 2
                    else:
                        negative_score += 1
        
        # Score calculation
        if positive_score > 0 or negative_score > 0:
            # Consider balance between positive and negative
            keyword_score = positive_score / (positive_score + negative_score + 1)
            # Keyword-based 50% + Review score 50%
            novelty_score = keyword_score * 0.5 + normalized_rating * 0.5
        else:
            # Use review score if no keywords found
            novelty_score = normalized_rating
        
        return min(MAX_SCORE, max(MIN_SCORE, novelty_score))
    
    def _generate_rationale(
        self,
        metadata: dict[str, Any],
        scores: dict[str, float],
    ) -> str:
        """Generate evaluation rationale in detailed text format.
        
        Args:
        ----
            metadata: Paper metadata
            scores: Calculated scores
            
        Returns:
        -------
            Evaluation rationale string
        """
        rating_avg = metadata.get("rating_avg")
        confidence_avg = metadata.get("confidence_avg")
        decision = metadata.get("decision", "N/A")
        num_reviews = len(metadata.get("reviews", []))
        
        parts = []
        
        # Basic information (review count and rating)
        if num_reviews > 0:
            parts.append(f"This paper received {num_reviews} review(s),")
            if rating_avg is not None:
                parts.append(f"with an average rating of {rating_avg:.2f}/10.")
            else:
                parts.append("but the rating score is not publicly available.")
        else:
            parts.append("This paper has not yet received reviews.")
        
        # Acceptance status
        decision_lower = decision.lower()
        if "oral" in decision_lower or "spotlight" in decision_lower:
            parts.append(f"The acceptance decision is \"{decision}\" and is particularly highly rated.")
        elif "accept" in decision_lower:
            parts.append(f"The acceptance decision is \"{decision}\".")
        elif "reject" in decision_lower:
            parts.append(f"Rejected ({decision}).")
        elif decision != "N/A":
            parts.append(f"Decision status: {decision}")
        
        # Score details
        parts.append(f"\n\n[Evaluation Score Details]")
        parts.append(f"Overall score: {scores['overall']:.3f}")
        parts.append(f"(Breakdown: Relevance {scores['relevance']:.3f}, ")
        parts.append(f"Novelty {scores['novelty']:.3f}, ")
        parts.append(f"Impact {scores['impact']:.3f})")
        
        # Reviewer confidence
        if confidence_avg is not None:
            confidence_desc = "very high" if confidence_avg >= 4.0 else "high" if confidence_avg >= 3.0 else "moderate"
            parts.append(f"\nReviewer confidence is {confidence_avg:.2f}/5 ({confidence_desc}).")
        
        return " ".join(parts)

