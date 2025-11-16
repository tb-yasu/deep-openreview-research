"""Paper Review Agent - Public Release Execution Program.

This script searches and evaluates papers from specified conferences,
ranks papers related to research interests, and generates reports.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

from app.paper_review_workflow.agent import create_graph, invoke_graph
from app.paper_review_workflow.models.state import (
    PaperReviewAgentInputState,
    EvaluationCriteria,
)
from app.paper_review_workflow.config import LLMConfig, LLMModel


def setup_logger(verbose: bool = False) -> None:
    """Set up logger."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if verbose else "INFO",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Paper Review Agent - Search and evaluate papers based on research interests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Specify research interests in natural language
  python run_deep_research.py --venue NeurIPS --year 2025 \\
    --research-description "I am interested in graph generation and its applications to drug discovery"

  # Specify with keyword list
  python run_deep_research.py --venue NeurIPS --year 2025 \\
    --research-interests "graph generation,drug discovery,molecular design"

  # With detailed settings
  python run_deep_research.py --venue NeurIPS --year 2025 \\
    --research-description "I am interested in graph generation and drug discovery applications" \\
    --top-k 50 --min-relevance-score 0.3 --model gpt-4o
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--venue",
        type=str,
        required=True,
        help="Conference name (e.g., NeurIPS, ICML, ICLR)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year (e.g., 2025)",
    )
    
    # Research interest specification (specify one of the two)
    research_group = parser.add_mutually_exclusive_group(required=True)
    research_group.add_argument(
        "--research-description",
        type=str,
        help="Describe research interests in natural language (recommended)",
    )
    research_group.add_argument(
        "--research-interests",
        type=str,
        help="Specify research interest keywords comma-separated (e.g., 'LLM,efficiency,fine-tuning')",
    )
    
    # Evaluation criteria
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        default=0.2,
        help="Minimum relevance score (0.0-1.0, default: 0.2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top papers for LLM evaluation (default: 100)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=15000,
        help="Maximum number of papers to search (default: 15000)",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        default=False,
        help="Include rejected papers in search (default: accepted papers only)",
    )
    parser.add_argument(
        "--focus-on-novelty",
        action="store_true",
        default=True,
        help="Prioritize novelty (default: True)",
    )
    parser.add_argument(
        "--focus-on-impact",
        action="store_true",
        default=True,
        help="Prioritize impact (default: True)",
    )
    
    # LLM settings
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="LLM model to use (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature parameter (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum LLM tokens (default: 1000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent LLM evaluations (default: 10, consider API rate limits)",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/outputs",
        help="Output directory (default: storage/outputs)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output filename (default: paper_review_report_{venue}_{year}.md)",
    )
    parser.add_argument(
        "--top-n-display",
        type=int,
        default=10,
        help="Number of papers to display in console (default: 10)",
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs",
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Skip LLM evaluation (keyword-based only)",
    )
    
    return parser.parse_args()


def get_llm_model(model_name: str) -> LLMModel:
    """Get LLMModel from model name."""
    model_map = {
        "gpt-4o": LLMModel.GPT4O,
        "gpt-4o-mini": LLMModel.GPT4O_MINI,
        "gpt-4-turbo": LLMModel.GPT4_TURBO,
        "gpt-5": LLMModel.GPT5,
        "gpt-5-mini": LLMModel.GPT5_MINI,
        "gpt-5-nano": LLMModel.GPT5_NANO,
    }
    return model_map.get(model_name, LLMModel.GPT4O_MINI)


def run_paper_review(args: argparse.Namespace) -> None:
    """Run paper review."""
    logger.info("=" * 100)
    logger.info("📚 Paper Review Agent")
    logger.info("=" * 100)
    
    try:
        # LLM configuration
        llm_config = LLMConfig(
            model=get_llm_model(args.model),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
        )
        
        # Create graph
        logger.info("🔧 Initializing workflow...")
        graph = create_graph(llm_config=llm_config)
        
        # Get research interests
        if args.research_description:
            research_description = args.research_description
            research_interests = []  # Will be auto-extracted
        else:
            research_description = None
            research_interests = [k.strip() for k in args.research_interests.split(",")]
        
        # Prepare input data
        input_data = PaperReviewAgentInputState(
            venue=args.venue,
            year=args.year,
            keywords=None,  # Use synonym matching
            max_papers=args.max_papers,
            accepted_only=not args.include_rejected,  # Default: accepted papers only
            evaluation_criteria=EvaluationCriteria(
                research_description=research_description,
                research_interests=research_interests,
                min_relevance_score=args.min_relevance_score,
                min_rating=None,  # Accepted papers are quality-assured
                enable_preliminary_llm_filter=False,
                top_k_papers=args.top_k if not args.no_llm_eval else None,
                focus_on_novelty=args.focus_on_novelty,
                focus_on_impact=args.focus_on_impact,
            ),
        )
        
        # Display execution conditions
        logger.info(f"\n📋 Execution Conditions:")
        logger.info(f"   Conference: {args.venue} {args.year}")
        if research_description:
            logger.info(f"   Research Interests: {research_description}")
        else:
            logger.info(f"   Keywords: {', '.join(research_interests)}")
        logger.info(f"   LLM Model: {args.model}")
        logger.info(f"   Min Relevance Score: {args.min_relevance_score}")
        logger.info(f"   Max Papers: {args.max_papers}")
        logger.info(f"   Search Scope: {'All papers (accepted & rejected)' if args.include_rejected else 'Accepted papers only'}")
        if not args.no_llm_eval:
            logger.info(f"   LLM Evaluation Target: Top {args.top_k} papers")
        else:
            logger.info(f"   LLM Evaluation: Skipped")
        
        # Execute agent
        logger.info("\n🚀 Running agent...")
        result = invoke_graph(
            graph=graph,
            input_data=input_data.model_dump(),
            config={
                "recursion_limit": 100,
                "thread_id": f"{args.venue}_{args.year}",
            },
        )
        
        # Get results
        papers = result.get("papers", [])
        evaluated_papers = result.get("evaluated_papers", [])
        ranked_papers = result.get("ranked_papers", [])
        llm_evaluated_papers = result.get("llm_evaluated_papers", [])
        re_ranked_papers = result.get("re_ranked_papers", [])
        top_papers = result.get("top_papers", [])
        paper_report = result.get("paper_report", "")
        synonyms = result.get("synonyms", {})
        
        # Display summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 Execution Results Summary")
        logger.info("=" * 100)
        logger.success(f"✓ Search: Found {len(papers)} papers")
        logger.success(f"✓ Evaluation: Evaluated {len(evaluated_papers)} papers")
        logger.success(f"✓ Ranking: Ranked {len(ranked_papers)} papers")
        if not args.no_llm_eval:
            logger.success(f"✓ LLM Evaluation: Evaluated {len(llm_evaluated_papers)} papers")
            logger.success(f"✓ Re-ranking: Re-ranked {len(re_ranked_papers)} papers")
        logger.success(f"✓ Selection: Selected {len(top_papers)} papers")
        
        # Display keywords and synonyms
        if synonyms:
            logger.info("\n" + "=" * 100)
            logger.info("🔑 Search Keywords and Synonyms")
            logger.info("=" * 100)
            for keyword, syns in synonyms.items():
                syns_display = ", ".join(syns[:5])
                if len(syns) > 5:
                    syns_display += f" +{len(syns) - 5} more"
                logger.info(f"📌 {keyword}")
                logger.info(f"   └ Synonyms: {syns_display}")
        
        # Display top N papers
        if top_papers and args.top_n_display > 0:
            logger.info("\n" + "=" * 100)
            logger.info(f"🏆 Top {args.top_n_display} Papers")
            logger.info("=" * 100)
            
            for paper in top_papers[:args.top_n_display]:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"【Rank #{paper['rank']}】 {paper['title']}")
                logger.info("")
                
                # Authors
                authors_list = paper['authors']
                authors_display = ', '.join(authors_list[:5])
                if len(authors_list) > 5:
                    authors_display += f" +{len(authors_list) - 5} more"
                logger.info(f"**Authors**: {authors_display}")
                
                # Keywords
                if paper.get('keywords'):
                    keywords_list = paper['keywords']
                    keywords_display = ', '.join(keywords_list[:8])
                    if len(keywords_list) > 8:
                        keywords_display += f" +{len(keywords_list) - 8} more"
                    logger.info(f"**Keywords**: {keywords_display}")
                logger.info("")
                
                # Abstract
                if paper.get('abstract'):
                    logger.info("#### Abstract")
                    logger.info("")
                    abstract = paper['abstract']
                    if len(abstract) > 400:
                        abstract = abstract[:400] + "..."
                    logger.info(abstract)
                    logger.info("")
                
                # Scores
                logger.info("#### Scores")
                logger.info("")
                if paper.get('final_score') is not None:
                    logger.info(f"| **Final Score**        | **{paper['final_score']:.3f}** |")
                if paper.get('overall_score') is not None:
                    logger.info(f"| Overall Score          | {paper['overall_score']:.3f} |")
                if paper.get('relevance_score') is not None:
                    logger.info(f"| 　├ Relevance          | {paper['relevance_score']:.3f} |")
                if paper.get('novelty_score') is not None:
                    logger.info(f"| 　├ Novelty            | {paper['novelty_score']:.3f} |")
                if paper.get('impact_score') is not None:
                    logger.info(f"| 　└ Impact             | {paper['impact_score']:.3f} |")
                if paper.get('llm_relevance_score') is not None:
                    logger.info(f"| AI Eval (Relevance)    | {paper['llm_relevance_score']:.3f} |")
                if paper.get('llm_novelty_score') is not None:
                    logger.info(f"| AI Eval (Novelty)      | {paper['llm_novelty_score']:.3f} |")
                if paper.get('llm_practical_score') is not None:
                    logger.info(f"| AI Eval (Practicality) | {paper['llm_practical_score']:.3f} |")
                if paper.get('rating_avg') is not None:
                    logger.info(f"| OpenReview Rating      | {paper['rating_avg']:.2f}/10 |")
                logger.info("")
                
                # OpenReview evaluation
                if not args.no_llm_eval:
                    logger.info("#### OpenReview Evaluation")
                    logger.info("")
                    rationale = paper.get('evaluation_rationale', '')
                    if rationale:
                        logger.info(rationale[:300] + ("..." if len(rationale) > 300 else ""))
                    else:
                        review_count = len(paper.get('reviews', []))
                        rating_info = f"average {paper['rating_avg']:.2f}/10" if paper.get('rating_avg') else "no rating"
                        decision = paper.get('decision', 'N/A')
                        logger.info(f"This paper received {review_count} reviews and achieved {rating_info}.")
                        logger.info(f"Acceptance decision: '{decision}'.")
                        
                        # Display presentation format (for NeurIPS, etc.)
                        if decision and decision != 'N/A':
                            decision_lower = decision.lower()
                            if "oral" in decision_lower:
                                logger.info("  └ 🎤 Presentation: Oral Presentation")
                            elif "spotlight" in decision_lower:
                                logger.info("  └ ✨ Presentation: Spotlight Presentation")
                            elif "poster" in decision_lower:
                                logger.info("  └ 📊 Presentation: Poster Presentation")
                    logger.info("")
                    
                    # Meta Review
                    if paper.get('meta_review') and paper['meta_review'].strip():
                        logger.info("#### 📋 Meta Review")
                        logger.info("")
                        meta_review = paper['meta_review']
                        if len(meta_review) > 200:
                            meta_review = meta_review[:200] + "..."
                        logger.info(meta_review)
                        logger.info("")
                    
                    # Review summary (display only first review)
                    reviews = paper.get('reviews', [])
                    if reviews and len(reviews) > 0:
                        first_review = reviews[0]
                        if first_review.get('summary') or first_review.get('strengths'):
                            logger.info("#### 📊 Review Highlights")
                            logger.info("")
                            if first_review.get('strengths'):
                                strengths = first_review['strengths']
                                logger.info("**Strengths:**")
                                logger.info(strengths[:150] + ("..." if len(strengths) > 150 else ""))
                    logger.info("")
                    
                    # AI evaluation
                    if paper.get('llm_rationale'):
                        logger.info("#### AI Evaluation (Content Analysis)")
                        logger.info("")
                        llm_rationale = paper['llm_rationale']
                        if len(llm_rationale) > 250:
                            llm_rationale = llm_rationale[:250] + "..."
                        logger.info(llm_rationale)
                        logger.info("")
                
                # Links
                logger.info("**🔗 Links**:")
                logger.info(f"- OpenReview: {paper['forum_url']}")
                if paper.get('pdf_url'):
                    logger.info(f"- PDF: {paper['pdf_url']}")
        
        # Save report to file
        if paper_report:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.output_file:
                output_file = output_dir / args.output_file
            else:
                output_file = output_dir / f"paper_review_report_{args.venue}_{args.year}.md"
            
            output_file.write_text(paper_report, encoding="utf-8")
            
            logger.info("\n" + "=" * 100)
            logger.success(f"📝 Report saved: {output_file}")
            logger.info(f"   File size: {len(paper_report) / 1024:.1f} KB")
            logger.info(f"   Lines: {len(paper_report.splitlines())} lines")
            logger.info("=" * 100)
        
        # Display errors if any
        errors = result.get("error_messages", [])
        if errors:
            logger.warning(f"\n⚠️  {len(errors)} error(s) occurred:")
            for i, error in enumerate(errors[:3], 1):
                logger.warning(f"  {i}. {error}")
            if len(errors) > 3:
                logger.warning(f"  ...and {len(errors) - 3} more")
        
        logger.info("\n✨ Processing completed!")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ An error occurred: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    setup_logger(verbose=args.verbose)
    run_paper_review(args)


if __name__ == "__main__":
    main()
