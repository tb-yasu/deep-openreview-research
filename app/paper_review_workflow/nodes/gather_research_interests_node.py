"""Node for gathering research interests interactively."""

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from app.paper_review_workflow.models.state import PaperReviewAgentState


class GatherResearchInterestsNode:
    """Node for collecting user's research interests interactively."""
    
    def __init__(self, min_keywords: int = 3):
        """Initialize GatherResearchInterestsNode.
        
        Args:
        ----
            min_keywords: Minimum number of keywords (default: 3)
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=500)
        self.min_keywords = min_keywords
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Collect research interests interactively.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Updated state dictionary
        """
        criteria = state.evaluation_criteria
        
        # Extract initial keywords
        if criteria.research_description:
            logger.info("Extracting keywords from research description...")
            initial_keywords = self._extract_keywords(criteria.research_description)
        else:
            initial_keywords = criteria.research_interests or []
        
        logger.info(f"\nExtracted keywords ({len(initial_keywords)} keywords):")
        for i, kw in enumerate(initial_keywords, 1):
            logger.info(f"  {i}. {kw}")
        
        # If keywords are insufficient, ask follow-up questions
        if len(initial_keywords) < self.min_keywords:
            logger.info(f"\nFewer than {self.min_keywords} keywords detected. Gathering additional information...\n")
            additional_keywords = self._ask_for_more_details(
                criteria.research_description or "",
                initial_keywords
            )
            
            # Merge and deduplicate
            all_keywords = list(set(initial_keywords + additional_keywords))
            
            logger.info(f"\nAdditional keywords: {additional_keywords}")
        else:
            all_keywords = initial_keywords
        
        # Final confirmation
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 Final Keyword List ({len(all_keywords)} keywords):")
        for i, kw in enumerate(all_keywords, 1):
            logger.info(f"  {i}. {kw}")
        logger.info(f"{'='*80}\n")
        
        # Return updated criteria
        updated_criteria = criteria.model_copy(deep=True)
        updated_criteria.research_interests = all_keywords
        
        return {
            "evaluation_criteria": updated_criteria,
        }
    
    def _extract_keywords(self, description: str) -> list[str]:
        """Extract keywords from natural language description.
        
        Args:
        ----
            description: Natural language description of research interests
            
        Returns:
        -------
            Extracted keyword list
        """
        try:
            prompt = f"""Extract key research topics from the following description.
Return 5-8 important keywords or phrases that represent the main research interests.

Description:
{description}

Return ONLY a JSON array of keywords, like:
["keyword1", "keyword2", "keyword3", ...]

Rules:
- Use lowercase
- Be specific and technical
- Include 5-8 keywords
- Focus on SPECIFIC application domains or methods
- EXCLUDE overly general ML/AI terms such as:
  × "machine learning", "deep learning", "artificial intelligence", "ai"
  × "neural networks", "data science", "data-driven approaches"
  × "predictive modeling", "optimization", "statistical analysis"
- Prefer domain-specific terms (e.g., "drug discovery" instead of "machine learning")

Good examples: ["drug discovery", "graph generation", "protein structure prediction"]
Bad examples: ["machine learning", "deep learning", "optimization"]
"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Parse JSON
            keywords = self._parse_json_response(response_text)
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}. Using empty list.")
            return []
    
    def _ask_for_more_details(
        self,
        initial_description: str,
        current_keywords: list[str]
    ) -> list[str]:
        """Ask follow-up questions to elicit more keywords (with suggestions).
        
        Args:
        ----
            initial_description: Initial description
            current_keywords: Current keyword list
            
        Returns:
        -------
            Additional keyword list
        """
        try:
            # Generate questions and suggestions using LLM
            question_prompt = f"""Based on this research description and current keywords, 
generate 2-3 specific follow-up questions in English with keyword suggestions.

Initial description: {initial_description}
Current keywords: {current_keywords}

For each question, provide:
- The question in English
- 3-5 example keywords (in English, lowercase) that would be relevant answers

Generate questions about:
- Specific methods or techniques they're interested in
- Application domains or use cases
- Related subfields or emerging topics

Return ONLY a JSON array of objects:
[
  {{
    "question": "Question 1?",
    "suggestions": ["keyword1", "keyword2", "keyword3"]
  }},
  {{
    "question": "Question 2?",
    "suggestions": ["keyword4", "keyword5"]
  }}
]
"""
            
            response = self.llm.invoke(question_prompt)
            questions_data = self._parse_json_response(response.content)
            
            # Display questions and collect answers
            logger.info("\nPlease answer the following questions:\n")
            answers = []
            
            for i, item in enumerate(questions_data, 1):
                if isinstance(item, dict):
                    question = item.get("question", "")
                    suggestions = item.get("suggestions", [])
                else:
                    # For compatibility, handle string case as well
                    question = str(item)
                    suggestions = []
                
                logger.info(f"{i}. {question}")
                
                # Display suggestions
                if suggestions:
                    suggestion_text = ", ".join(suggestions[:5])
                    logger.info(f"   [Examples: {suggestion_text}]")
                
                try:
                    answer = input("   Answer: ")
                    if answer.strip():
                        answers.append(answer)
                    else:
                        logger.info("   (Skipped)")
                except (EOFError, KeyboardInterrupt):
                    logger.info("\n(Skipped)")
                    break
            
            # Extract additional keywords from answers
            if answers:
                combined_answers = " ".join(answers)
                additional_prompt = f"""Extract additional research keywords from these answers:

{combined_answers}

Return 3-5 additional keywords as JSON array.
Avoid duplicating: {current_keywords}
Use lowercase and be specific.
"""
                
                response = self.llm.invoke(additional_prompt)
                additional_keywords = self._parse_json_response(response.content)
                
                return additional_keywords
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to ask for more details: {e}")
            return []
    
    def _parse_json_response(self, response_text: str) -> list:
        """Parse JSON array from LLM response.
        
        Args:
        ----
            response_text: LLM response text
            
        Returns:
        -------
            Parsed list (list of strings or list of objects)
        """
        try:
            # Extract JSON block
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse JSON
            result = json.loads(json_str)
            
            if isinstance(result, list):
                # Return as-is if list elements are dictionaries
                if result and isinstance(result[0], dict):
                    return result
                # If strings, lowercase and return
                return [str(item).lower().strip() for item in result]
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response: {response_text[:200]}...")
            return []
