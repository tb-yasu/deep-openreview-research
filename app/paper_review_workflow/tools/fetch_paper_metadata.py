"""Tool for fetching detailed paper metadata using OpenReview API."""

import json
from typing import Any

import openreview
from langchain_core.tools import tool
from loguru import logger


@tool
def fetch_paper_metadata(paper_id: str) -> str:
    """Fetch detailed metadata for specified paper ID using OpenReview API.

    Args:
    ----
        paper_id (str): OpenReview paper ID (e.g., 'abc123def456')

    Returns:
    -------
        str: JSON of detailed paper metadata. Includes:
            - id: Paper ID
            - title: Paper title
            - authors: Author list
            - abstract: Abstract
            - keywords: Keyword list
            - reviews: List of review information
            - rating_avg: Average rating score
            - confidence_avg: Average confidence score
            - decision: Acceptance/rejection decision
            - pdf_url: PDF URL
            - forum_url: Forum URL

    """
    try:
        # Initialize OpenReview API client
        client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

        # Get paper information
        logger.info(f"Fetching metadata for paper: {paper_id}")
        note = client.get_note(paper_id)

        # Get review information
        reviews = client.get_notes(forum=paper_id, invitation=".*Review$")
        
        # Aggregate evaluation scores
        ratings: list[float] = []
        confidences: list[float] = []
        review_list: list[dict[str, Any]] = []
        
        for review in reviews:
            rating = review.content.get("rating", {})
            confidence = review.content.get("confidence", {})
            
            if isinstance(rating, dict) and "value" in rating:
                try:
                    # Extract numeric value from format like "8: accept"
                    rating_value = float(str(rating["value"]).split(":")[0].strip())
                    ratings.append(rating_value)
                except (ValueError, IndexError):
                    pass
            
            if isinstance(confidence, dict) and "value" in confidence:
                try:
                    confidence_value = float(str(confidence["value"]).split(":")[0].strip())
                    confidences.append(confidence_value)
                except (ValueError, IndexError):
                    pass
            
            summary = review.content.get("summary", {})
            summary_value = summary.get("value", "") if isinstance(summary, dict) else str(summary)
            
            strengths = review.content.get("strengths", {})
            strengths_value = strengths.get("value", "") if isinstance(strengths, dict) else str(strengths)
            
            weaknesses = review.content.get("weaknesses", {})
            weaknesses_value = weaknesses.get("value", "") if isinstance(weaknesses, dict) else str(weaknesses)
            
            review_list.append({
                "rating": str(rating.get("value", "N/A")) if isinstance(rating, dict) else "N/A",
                "confidence": str(confidence.get("value", "N/A")) if isinstance(confidence, dict) else "N/A",
                "summary": summary_value,
                "strengths": strengths_value,
                "weaknesses": weaknesses_value,
            })

        # Get acceptance decision
        decisions = client.get_notes(forum=paper_id, invitation=".*Decision$")
        decision = "N/A"
        if decisions:
            decision_content = decisions[0].content.get("decision", {})
            decision = decision_content.get("value", "N/A") if isinstance(decision_content, dict) else str(decision_content)

        # Build metadata
        title = note.content.get("title", {})
        title_value = title.get("value", "") if isinstance(title, dict) else str(title)
        
        authors = note.content.get("authors", {})
        authors_value = authors.get("value", []) if isinstance(authors, dict) else []
        
        abstract = note.content.get("abstract", {})
        abstract_value = abstract.get("value", "") if isinstance(abstract, dict) else str(abstract)
        
        keywords = note.content.get("keywords", {})
        keywords_value = keywords.get("value", []) if isinstance(keywords, dict) else []
        
        metadata = {
            "id": note.id,
            "title": title_value,
            "authors": authors_value,
            "abstract": abstract_value,
            "keywords": keywords_value,
            "reviews": review_list,
            "rating_avg": sum(ratings) / len(ratings) if ratings else None,
            "confidence_avg": sum(confidences) / len(confidences) if confidences else None,
            "decision": decision,
            "pdf_url": f"https://openreview.net/pdf?id={note.id}",
            "forum_url": f"https://openreview.net/forum?id={note.id}",
        }

        logger.info(f"Fetched metadata for paper: {metadata['title']}")
        return json.dumps(metadata, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"Error fetching paper metadata: {e!s}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

