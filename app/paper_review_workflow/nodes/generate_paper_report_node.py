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
        lines.append(f"- **Papers Found**: {len(state.papers)} papers")
        lines.append(f"- **Papers Evaluated**: {len(state.evaluated_papers)} papers")
        lines.append(f"- **Papers Ranked**: {len(state.ranked_papers)} papers")
        lines.append("")
        
        # Evaluation criteria
        criteria = state.evaluation_criteria
        lines.append("## Evaluation Criteria")
        lines.append("")
        lines.append(f"- **Research Interests**: {', '.join(criteria.research_interests)}")
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
            
            # Score display (unified LLM evaluation version)
            lines.append("#### Scores")
            lines.append("")
            lines.append(f"| Item | Score |")
            lines.append(f"|------|-------|")
            
            # Overall score (weighted average of 4 scores)
            overall_score = paper.get('overall_score') if isinstance(paper, dict) else getattr(paper, 'overall_score', None)
            if overall_score is not None:
                lines.append(f"| **Overall Score** | **{overall_score:.3f}** |")
            
            # AI evaluation detailed scores
            relevance_score = paper.get('relevance_score') if isinstance(paper, dict) else getattr(paper, 'relevance_score', None)
            if relevance_score is not None:
                lines.append(f"| 　├ Relevance | {relevance_score:.3f} |")
            
            novelty_score = paper.get('novelty_score') if isinstance(paper, dict) else getattr(paper, 'novelty_score', None)
            if novelty_score is not None:
                lines.append(f"| 　├ Novelty | {novelty_score:.3f} |")
            
            impact_score = paper.get('impact_score') if isinstance(paper, dict) else getattr(paper, 'impact_score', None)
            if impact_score is not None:
                lines.append(f"| 　├ Impact | {impact_score:.3f} |")
            
            practicality_score = paper.get('practicality_score') if isinstance(paper, dict) else getattr(paper, 'practicality_score', None)
            if practicality_score is not None:
                lines.append(f"| 　└ Practicality | {practicality_score:.3f} |")
            
            # OpenReview average rating
            rating_avg = paper.get('rating_avg') if isinstance(paper, dict) else getattr(paper, 'rating_avg', None)
            if rating_avg is not None:
                lines.append(f"| OpenReview Rating | {rating_avg:.2f}/10 |")
            lines.append("")
            
            # Acceptance decision and presentation format
            decision = paper.get('decision') if isinstance(paper, dict) else getattr(paper, 'decision', None)
            if decision and decision != "N/A":
                lines.append(f"**Acceptance Decision**: {decision}")
                
                # Extract presentation format (for NeurIPS, etc.)
                decision_lower = decision.lower()
                if "oral" in decision_lower:
                    lines.append("  - 🎤 **Presentation**: Oral Presentation")
                elif "spotlight" in decision_lower:
                    lines.append("  - ✨ **Presentation**: Spotlight Presentation")
                elif "poster" in decision_lower:
                    lines.append("  - 📊 **Presentation**: Poster Presentation")
                lines.append("")
            
            # Authors
            authors = paper.get('authors') if isinstance(paper, dict) else paper.authors
            if authors:
                authors_display = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_display += f" +{len(authors) - 5} more"
                lines.append(f"**Authors**: {authors_display}")
                lines.append("")
            
            # Keywords
            keywords = paper.get('keywords') if isinstance(paper, dict) else paper.keywords
            if keywords:
                lines.append(f"**Keywords**: {', '.join(keywords[:8])}")
                lines.append("")
            
            # Abstract (full text, as independent section)
            abstract = paper.get('abstract') if isinstance(paper, dict) else getattr(paper, 'abstract', '')
            if abstract and abstract.strip():
                lines.append("#### Abstract")
                lines.append("")
                lines.append(abstract)
                lines.append("")
            
            # AI Evaluation (Unified LLM Evaluation)
            ai_rationale = paper.get('ai_rationale') if isinstance(paper, dict) else getattr(paper, 'ai_rationale', None)
            if ai_rationale and ai_rationale.strip():
                lines.append("#### 🤖 AI Evaluation")
                lines.append("")
                lines.append(ai_rationale)
                lines.append("")
            
            # Review Summary
            review_summary = paper.get('review_summary') if isinstance(paper, dict) else getattr(paper, 'review_summary', None)
            if review_summary and review_summary.strip():
                lines.append("#### 📊 Review Summary")
                lines.append("")
                lines.append(review_summary)
                lines.append("")
            
            # Field Insights
            field_insights = paper.get('field_insights') if isinstance(paper, dict) else getattr(paper, 'field_insights', None)
            if field_insights and field_insights.strip():
                lines.append("#### 🔍 Evaluation Data Sources")
                lines.append("")
                lines.append(field_insights)
                lines.append("")
            
            # Meta Review (Area Chair Summary)
            meta_review = paper.get('meta_review') if isinstance(paper, dict) else getattr(paper, 'meta_review', None)
            if meta_review and meta_review.strip():
                lines.append("#### 📋 Meta Review")
                lines.append("")
                # Limit length if too long (first ~800 characters)
                if len(meta_review) > 800:
                    lines.append(meta_review[:800] + "...")
                else:
                    lines.append(meta_review)
                lines.append("")
            
            # Decision detailed comments
            decision_comment = paper.get('decision_comment') if isinstance(paper, dict) else getattr(paper, 'decision_comment', None)
            if decision_comment and decision_comment.strip():
                lines.append("#### 📝 Acceptance Reason")
                lines.append("")
                # Limit length if too long
                if len(decision_comment) > 600:
                    lines.append(decision_comment[:600] + "...")
                else:
                    lines.append(decision_comment)
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
            
            # Display average review scores
            reviews = paper.get('reviews') if isinstance(paper, dict) else getattr(paper, 'reviews', [])
            if reviews and len(reviews) > 0:
                lines.append("#### 📊 Average Review Scores")
                lines.append("")
                
                # Collect numeric fields
                score_fields = {}
                for review in reviews:
                    for key, value in review.items():
                        # Only numeric or convertible to numeric fields
                        if key in ['summary', 'strengths', 'weaknesses', 'detailed_comments', 
                                  'main_review', 'review_text', 'comments', 'strengths_and_weaknesses']:
                            continue  # Skip text fields
                        
                        try:
                            # Try to convert to numeric
                            if isinstance(value, (int, float)):
                                numeric_value = float(value)
                            elif isinstance(value, str) and value.strip():
                                # Handle formats like "3.5/5"
                                if '/' in value:
                                    numeric_value = float(value.split('/')[0].strip())
                                else:
                                    numeric_value = float(value)
                            else:
                                continue
                            
                            if key not in score_fields:
                                score_fields[key] = []
                            score_fields[key].append(numeric_value)
                        except (ValueError, TypeError):
                            continue
                
                # Calculate and display averages
                if score_fields:
                    lines.append("| Item | Average | # Reviews |")
                    lines.append("|------|---------|-----------|")
                    
                    # Display rating and confidence first
                    priority_fields = ['rating', 'overall_recommendation', 'confidence']
                    for field in priority_fields:
                        if field in score_fields:
                            avg_value = sum(score_fields[field]) / len(score_fields[field])
                            count = len(score_fields[field])
                            field_name = self._translate_field_name(field)
                            lines.append(f"| {field_name} | {avg_value:.2f} | {count} |")
                    
                    # Display other fields in alphabetical order
                    other_fields = sorted([f for f in score_fields.keys() if f not in priority_fields])
                    for field in other_fields:
                        avg_value = sum(score_fields[field]) / len(score_fields[field])
                        count = len(score_fields[field])
                        field_name = self._translate_field_name(field)
                        lines.append(f"| {field_name} | {avg_value:.2f} | {count} |")
                    
                    lines.append("")
                    lines.append(f"*Aggregated from {len(reviews)} review(s)*")
                    lines.append("")
            
            # Author Final Remarks
            author_remarks = paper.get('author_remarks') if isinstance(paper, dict) else getattr(paper, 'author_remarks', None)
            if author_remarks and author_remarks.strip():
                lines.append("#### 💬 Author Comments")
                lines.append("")
                # Limit length if too long
                if len(author_remarks) > 400:
                    lines.append(author_remarks[:400] + "...")
                else:
                    lines.append(author_remarks)
                lines.append("")
            
            # LLM evaluation rationale
            llm_rationale = paper.get('llm_rationale') if isinstance(paper, dict) else getattr(paper, 'llm_rationale', None)
            if llm_rationale:
                lines.append("#### AI Evaluation (Content Analysis)")
                lines.append("")
                lines.append(llm_rationale)
                lines.append("")
            
            # Links
            forum_url = paper.get('forum_url') if isinstance(paper, dict) else paper.forum_url
            pdf_url = paper.get('pdf_url') if isinstance(paper, dict) else paper.pdf_url
            lines.append(f"**🔗 Links**:")
            lines.append(f"- [OpenReview]({forum_url})")
            lines.append(f"- [PDF]({pdf_url})")
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
