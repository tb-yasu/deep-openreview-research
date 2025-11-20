"""Tool for searching papers using OpenReview API."""

import json
from pathlib import Path
from typing import Any

import openreview
from langchain_core.tools import tool
from loguru import logger

from app.paper_review_workflow.tools.cache_utils import (
    get_cache_key,
    get_cached_data,
    save_to_cache,
)

from app.paper_review_workflow.tools.cache_manager import CacheManager

# Global cache manager
_cache_manager = CacheManager(cache_dir="storage/cache", ttl_hours=24)


def is_accepted(paper: dict[str, Any]) -> bool:
    """Check if paper is accepted.
    
    Args:
    ----
        paper: Dictionary of paper information
        
    Returns:
    -------
        bool: True if accepted, False otherwise
    """
    decision = paper.get("decision", "").lower()
    return "accept" in decision


@tool
def search_papers(
    venue: str,
    year: int,
    keywords: str | None = None,
    max_results: int = 100,
    accepted_only: bool = True,
) -> str:
    """Search for papers from specified conference/year using OpenReview API.

    Args:
    ----
        venue (str): Conference name (e.g., 'NeurIPS', 'ICML', 'ICLR')
        year (int): Conference year (e.g., 2024)
        keywords (str, optional): Search keywords (filter by paper title or abstract)
        max_results (int): Maximum number of results (default: 100)
        accepted_only (bool): Search only accepted papers (default: True)

    Returns:
    -------
        str: JSON list of paper information. Each paper includes:
            - id: Paper ID
            - title: Paper title
            - authors: Author list
            - abstract: Abstract
            - keywords: Keyword list
            - venue: Conference name
            - year: Year
            - pdf_url: PDF URL
            - forum_url: Forum URL

    """
    try:
        # First check local cache for all papers
        data_dir = Path(f"storage/papers_data/{venue}_{year}")
        papers_file = data_dir / "all_papers.json"
        
        if papers_file.exists():
            logger.info(f"Loading from local papers data: {papers_file}")
            all_papers = json.loads(papers_file.read_text(encoding="utf-8"))
            
            # Filter by keywords and acceptance status
            filtered_papers: list[dict[str, Any]] = []
            skipped_rejected = 0
            
            for paper in all_papers:
                # Filter only accepted papers (if accepted_only=True)
                if accepted_only and not is_accepted(paper):
                    skipped_rejected += 1
                    continue
                
                # Filter by keywords
                if keywords:
                    title_match = keywords.lower() in paper["title"].lower()
                    abstract_match = keywords.lower() in paper["abstract"].lower()
                    if not (title_match or abstract_match):
                        continue
                
                filtered_papers.append(paper)
                
                if len(filtered_papers) >= max_results:
                    break
            
            filter_msg = f"Found {len(filtered_papers)} papers (filtered from {len(all_papers)} total)"
            if accepted_only and skipped_rejected > 0:
                filter_msg += f", skipped {skipped_rejected} rejected papers"
            logger.info(filter_msg)
            
            return json.dumps(filtered_papers, ensure_ascii=False, indent=2)
        
        # If no local cache, check regular cache
        logger.info("No local papers data found. Checking temporary cache...")
        cache_key_params = {
            "venue": venue,
            "year": year,
            "keywords": keywords,
            "max_results": max_results,
        }
        cached_result = _cache_manager.get(prefix="search_papers", **cache_key_params)
        
        if cached_result:
            logger.info(f"Using temporary cached results for {venue} {year} (keywords: {keywords})")
            return cached_result
        
        # If no cache, fetch from API
        logger.warning(f"No cache found. Downloading from OpenReview API...")
        logger.warning(f"TIP: Run 'python fetch_all_papers.py' to download all papers once")
        
        # Initialize OpenReview API client (no authentication)
        client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

        # Construct conference invitation ID
        # Example: NeurIPS.cc/2024/Conference/-/Submission
        venue_id = f"{venue}.cc/{year}/Conference"
        
        # Get accepted papers
        logger.info(f"Searching papers from {venue} {year}...")
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/Submission",
            details="replies",
        )

        papers: list[dict[str, Any]] = []
        for submission in submissions:
            # Keyword filtering
            if keywords:
                title = submission.content.get("title", {})
                title_value = title.get("value", "") if isinstance(title, dict) else str(title)
                
                abstract = submission.content.get("abstract", {})
                abstract_value = abstract.get("value", "") if isinstance(abstract, dict) else str(abstract)
                
                title_match = keywords.lower() in title_value.lower()
                abstract_match = keywords.lower() in abstract_value.lower()
                
                if not (title_match or abstract_match):
                    continue

            # Extract paper information
            title = submission.content.get("title", {})
            title_value = title.get("value", "") if isinstance(title, dict) else str(title)
            
            authors = submission.content.get("authors", {})
            authors_value = authors.get("value", []) if isinstance(authors, dict) else []
            
            abstract = submission.content.get("abstract", {})
            abstract_value = abstract.get("value", "") if isinstance(abstract, dict) else str(abstract)
            
            keywords_field = submission.content.get("keywords", {})
            keywords_value = keywords_field.get("value", []) if isinstance(keywords_field, dict) else []
            
            paper_info = {
                "id": submission.id,
                "title": title_value,
                "authors": authors_value,
                "abstract": abstract_value,
                "keywords": keywords_value,
                "venue": venue,
                "year": year,
                "pdf_url": f"https://openreview.net/pdf?id={submission.id}",
                "forum_url": f"https://openreview.net/forum?id={submission.id}",
            }
            papers.append(paper_info)
            
            # Stop when max_results is reached
            if len(papers) >= max_results:
                break

        logger.info(f"Found {len(papers)} papers from {venue} {year}")
        result = json.dumps(papers, ensure_ascii=False, indent=2)
        
        # Save to cache
        _cache_manager.set(result, prefix="search_papers", **cache_key_params)
        
        return result

    except Exception as e:
        error_msg = f"Error searching papers: {e!s}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

