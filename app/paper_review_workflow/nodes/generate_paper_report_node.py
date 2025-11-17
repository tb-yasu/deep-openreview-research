"""Node for generating paper review report."""

from typing import Any
from datetime import datetime

from loguru import logger

from app.paper_review_workflow.models.state import PaperReviewAgentState


class GeneratePaperReportNode:
    """Node for generating paper review report."""
    
    def __init__(self) -> None:
        """Initialize GeneratePaperReportNode."""
        pass
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute report generation.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Updated state dictionary
        """
        logger.info("Generating paper review report...")
        
        report = self._generate_markdown_report(state)
        
        logger.success("Paper review report generated successfully")
        
        return {
            "paper_report": report,
        }
    
    def _generate_markdown_report(self, state: PaperReviewAgentState) -> str:
        """Generate Markdown report.
        
        Args:
        ----
            state: Current state
            
        Returns:
        -------
            Report in Markdown format
        """
        lines = []
        
        # Title
        lines.append("# Paper Review Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # Search criteria
        lines.append("## Search Criteria")
        lines.append("")
        lines.append(f"- **Conference**: {state.venue} {state.year}")
        lines.append(f"- **Keywords**: {state.keywords or 'Not specified'}")
        
        # Add research description
        criteria = state.evaluation_criteria
        if criteria.research_description:
            lines.append(f"- **Research Description**: {criteria.research_description}")
        
        lines.append("")
        
        # Hit counts details
        lines.append("## Hit Counts")
        lines.append("")
        lines.append(f"- **Total Papers**: {len(state.papers)} papers")
        lines.append(f"- **Papers Evaluated**: {len(state.evaluated_papers)} papers")
        lines.append(f"- **Papers Ranked**: {len(state.ranked_papers)} papers")
        if state.top_papers:
            lines.append(f"- **Final Selected Papers**: {len(state.top_papers)} papers")
        lines.append("")
        
        # Evaluation criteria
        lines.append("## Evaluation Criteria")
        lines.append("")
        lines.append(f"- **Research Interest Keywords**: {', '.join(criteria.research_interests)}")
        lines.append(f"- **Min Relevance Score**: {criteria.min_relevance_score}")
        if criteria.min_rating:
            lines.append(f"- **Min Review Rating**: {criteria.min_rating}/10")
        lines.append(f"- **Focus on Novelty**: {'Yes' if criteria.focus_on_novelty else 'No'}")
        lines.append(f"- **Focus on Impact**: {'Yes' if criteria.focus_on_impact else 'No'}")
        lines.append("")
        
        # Keywords and synonyms
        if state.synonyms:
            lines.append("## Keywords and Synonyms")
            lines.append("")
            lines.append("LLM-generated synonyms were used to search for papers matching each keyword.")
            lines.append("")
            for keyword, syns in state.synonyms.items():
                lines.append(f"### {keyword}")
                lines.append("")
                if syns:
                    lines.append("**Synonyms**:")
                    for syn in syns:
                        lines.append(f"- {syn}")
                else:
                    lines.append("No synonyms (using original keyword only)")
                lines.append("")
        
        # Statistics
        if state.ranked_papers:
            scores = [p.overall_score for p in state.ranked_papers if p.overall_score]
            ratings = [p.rating_avg for p in state.ranked_papers if p.rating_avg]
            
            lines.append("## Statistics")
            lines.append("")
            if scores:
                lines.append(f"- **Average Overall Score**: {sum(scores) / len(scores):.3f}")
                lines.append(f"- **Highest Score**: {max(scores):.3f}")
                lines.append(f"- **Lowest Score**: {min(scores):.3f}")
            if ratings:
                lines.append(f"- **Average Review Rating**: {sum(ratings) / len(ratings):.2f}/10")
            lines.append("")
        
        # Top papers (from top_papers if available, otherwise from ranked_papers)
        lines.append("## Top Papers")
        lines.append("")
        
        # Use top_papers if available (LLM evaluated), otherwise use ranked_papers
        papers_to_display = state.top_papers if state.top_papers else state.ranked_papers[:10]
        
        for i, paper_data in enumerate(papers_to_display[:20], 1):  # Top 20 papers
            # Handle both dictionary and EvaluatedPaper object cases
            if isinstance(paper_data, dict):
                paper = paper_data
                rank = paper.get('rank', i)
            else:
                paper = paper_data
                rank = getattr(paper, 'rank', i)
            
            # Get title (compatible with both dict and object)
            title = paper.get('title') if isinstance(paper, dict) else paper.title
            lines.append(f"### {rank}. {title}")
            lines.append("")
            
            # TL;DR (3-line summary)
            ai_rationale = paper.get('ai_rationale') if isinstance(paper, dict) else getattr(paper, 'ai_rationale', None)
            review_summary = paper.get('review_summary') if isinstance(paper, dict) else getattr(paper, 'review_summary', None)
            decision = paper.get('decision') if isinstance(paper, dict) else getattr(paper, 'decision', None)
            
            lines.append("#### 🎯 TL;DR")
            lines.append("")
            
            # Extract key points from AI evaluation (first ~150 chars)
            if ai_rationale and ai_rationale.strip():
                tldr_text = ai_rationale[:150].split('.')[0] + '.'
                lines.append(f"- **Proposal & Strengths**: {tldr_text}")
            
            # Extract evaluation from review summary
            if review_summary and review_summary.strip():
                review_tldr = review_summary[:100].split('.')[0] + '.'
                lines.append(f"- **Review Evaluation**: {review_tldr}")
            
            # Decision
            if decision and decision != "N/A":
                decision_lower = decision.lower()
                if "oral" in decision_lower:
                    lines.append(f"- **Decision**: Accepted (🎤 Oral)")
                elif "spotlight" in decision_lower:
                    lines.append(f"- **Decision**: Accepted (✨ Spotlight)")
                elif "poster" in decision_lower:
                    lines.append(f"- **Decision**: Accepted (📊 Poster)")
                elif "accept" in decision_lower:
                    lines.append(f"- **Decision**: Accepted")
                else:
                    lines.append(f"- **Decision**: {decision}")
            
            lines.append("")
            
            # Score in one line (simplified)
            overall_score = paper.get('overall_score') if isinstance(paper, dict) else getattr(paper, 'overall_score', None)
            relevance_score = paper.get('relevance_score') if isinstance(paper, dict) else getattr(paper, 'relevance_score', None)
            novelty_score = paper.get('novelty_score') if isinstance(paper, dict) else getattr(paper, 'novelty_score', None)
            impact_score = paper.get('impact_score') if isinstance(paper, dict) else getattr(paper, 'impact_score', None)
            practicality_score = paper.get('practicality_score') if isinstance(paper, dict) else getattr(paper, 'practicality_score', None)
            rating_avg = paper.get('rating_avg') if isinstance(paper, dict) else getattr(paper, 'rating_avg', None)
            
            score_parts = []
            if overall_score is not None:
                score_parts.append(f"**Overall: {overall_score:.3f}**")
            if relevance_score is not None:
                score_parts.append(f"Relevance: {relevance_score:.2f}")
            if novelty_score is not None:
                score_parts.append(f"Novelty: {novelty_score:.2f}")
            if impact_score is not None:
                score_parts.append(f"Impact: {impact_score:.2f}")
            if practicality_score is not None:
                score_parts.append(f"Practicality: {practicality_score:.2f}")
            if rating_avg is not None:
                score_parts.append(f"OpenReview: {rating_avg:.2f}/10")
            
            if score_parts:
                lines.append("**Scores**: " + " | ".join(score_parts))
                lines.append("")
            
            # Authors and Keywords (concise)
            authors = paper.get('authors') if isinstance(paper, dict) else paper.authors
            keywords = paper.get('keywords') if isinstance(paper, dict) else paper.keywords
            
            info_parts = []
            if authors:
                authors_display = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_display += f" +{len(authors) - 3} more"
                info_parts.append(f"**Authors**: {authors_display}")
            if keywords:
                info_parts.append(f"**Keywords**: {', '.join(keywords[:5])}")
            
            if info_parts:
                lines.append(" | ".join(info_parts))
                lines.append("")
            
            # Abstract (shortened - first 300 characters or 3 sentences)
            abstract = paper.get('abstract') if isinstance(paper, dict) else getattr(paper, 'abstract', '')
            if abstract and abstract.strip():
                lines.append("#### Abstract")
                lines.append("")
                # Display first 300 characters or first 3 sentences
                abstract_short = abstract[:300]
                sentences = abstract_short.split('.')
                if len(sentences) > 3:
                    abstract_short = '.'.join(sentences[:3]) + '....'
                elif len(abstract) > 300:
                    abstract_short += '...'
                lines.append(abstract_short)
                lines.append("")
            
            # Evaluation Highlights (merged AI evaluation + Review summary)
            if (ai_rationale and ai_rationale.strip()) or (review_summary and review_summary.strip()):
                lines.append("#### 📊 Evaluation Highlights")
                lines.append("")
                
                if ai_rationale and ai_rationale.strip():
                    lines.append("**AI Analysis**:")
                    lines.append(ai_rationale)
                    lines.append("")
                
                if review_summary and review_summary.strip():
                    lines.append("**Review Summary**:")
                    lines.append(review_summary)
                    lines.append("")
            
            # Review details (Strengths/Weaknesses) - Commented out (hidden per user request)
            # reviews = paper.get('reviews') if isinstance(paper, dict) else getattr(paper, 'reviews', [])
            # if reviews and len(reviews) > 0:
            #     lines.append("#### 📊 Review Details")
            #     lines.append("")
            #     for review_idx, review in enumerate(reviews[:3], 1):  # Max 3 reviews
            #         review_rating = review.get('rating', 'N/A')
            #         review_confidence = review.get('confidence', 'N/A')
            #         lines.append(f"**Review {review_idx}** (Rating: {review_rating}, Confidence: {review_confidence})")
            #         lines.append("")
            #         
            #         # Summary
            #         summary = review.get('summary', '')
            #         if summary and summary.strip():
            #             lines.append("**Summary:**")
            #             summary_text = summary[:300] + ("..." if len(summary) > 300 else "")
            #             lines.append(summary_text)
            #             lines.append("")
            #         
            #         # Strengths
            #         strengths = review.get('strengths', '')
            #         if strengths and strengths.strip():
            #             lines.append("**Strengths:**")
            #             strengths_text = strengths[:300] + ("..." if len(strengths) > 300 else "")
            #             lines.append(strengths_text)
            #             lines.append("")
            #         
            #         # Weaknesses
            #         weaknesses = review.get('weaknesses', '')
            #         if weaknesses and weaknesses.strip():
            #             lines.append("**Weaknesses:**")
            #             weaknesses_text = weaknesses[:300] + ("..." if len(weaknesses) > 300 else "")
            #             lines.append(weaknesses_text)
            #             lines.append("")
            #     
            #     if len(reviews) > 3:
            #         lines.append(f"*{len(reviews) - 3} more review(s) omitted*")
            #         lines.append("")
            
            # Links (simplified)
            forum_url = paper.get('forum_url') if isinstance(paper, dict) else paper.forum_url
            pdf_url = paper.get('pdf_url') if isinstance(paper, dict) else paper.pdf_url
            lines.append(f"🔗 [OpenReview]({forum_url}) | [PDF]({pdf_url})")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _translate_field_name(self, field: str) -> str:
        """Translate review field name to English with original field name in parentheses.
        
        Args:
        ----
            field: English field name
            
        Returns:
        -------
            Translated field name with original in parentheses
        """
        translations = {
            # Basic scores
            'rating': 'Overall Rating',
            'overall_recommendation': 'Overall Recommendation',
            'confidence': 'Confidence',
            'score': 'Score',
            'recommendation': 'Recommendation',
            
            # ICLR/NeurIPS/ICML common fields
            'soundness': 'Soundness',
            'presentation': 'Presentation',
            'contribution': 'Contribution',
            'originality': 'Originality',
            'quality': 'Quality',
            'clarity': 'Clarity',
            'significance': 'Significance',
            
            # ICML specific fields
            'experimental_designs_or_analyses': 'Experimental Design',
            'methods_and_evaluation_criteria': 'Methods & Evaluation',
            'reproducibility': 'Reproducibility',
            'claims_and_evidence': 'Claims & Evidence',
            'impact': 'Impact',
            'novelty': 'Novelty',
            
            # Other
            'technical_novelty_and_significance': 'Technical Novelty & Significance',
            'potential_for_real_world_impact': 'Real-world Impact Potential',
            'ethical_considerations': 'Ethical Considerations',
        }
        
        return translations.get(field, field.replace('_', ' ').title())
