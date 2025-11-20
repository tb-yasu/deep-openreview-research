"""Tool for analyzing paper citations using Semantic Scholar API."""

import json
from typing import Any

import requests
from langchain_core.tools import tool
from loguru import logger


@tool
def analyze_citations(
    paper_title: str,
    doi: str | None = None,
    max_citations: int = 20,
    max_references: int = 20,
) -> str:
    """Analyze paper citation information using Semantic Scholar API.

    Args:
    ----
        paper_title (str): Paper title
        doi (str, optional): DOI (if available)
        max_citations (int): Maximum number of citing papers to retrieve (default: 20)
        max_references (int): Maximum number of references to retrieve (default: 20)

    Returns:
    -------
        str: JSON of citation analysis results. Includes:
            - paper_id: Paper ID on Semantic Scholar
            - title: Paper title
            - citation_count: Citation count
            - reference_count: Reference count
            - influential_citation_count: Influential citation count
            - year: Publication year
            - citations: List of citing papers (max max_citations)
            - references: List of references (max max_references)
            - citation_velocity: Average citations per year

    """
    try:
        base_url = "https://api.semanticscholar.org/graph/v1"
        
        # Search for paper
        logger.info(f"Analyzing citations for: {paper_title}")
        if doi:
            # Get directly if DOI is available
            search_url = f"{base_url}/paper/DOI:{doi}"
        else:
            # Search by title
            search_url = f"{base_url}/paper/search"
            params = {"query": paper_title, "limit": 1}
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            search_results = response.json()
            
            if not search_results.get("data"):
                return json.dumps(
                    {"error": "Paper not found in Semantic Scholar"},
                    ensure_ascii=False,
                )
            
            paper_id = search_results["data"][0]["paperId"]
            search_url = f"{base_url}/paper/{paper_id}"

        # Get paper details (including citation information)
        params = {
            "fields": "paperId,title,year,citationCount,referenceCount,influentialCitationCount,citations,citations.title,citations.year,citations.authors,references,references.title,references.year,references.authors"
        }
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        paper_data = response.json()

        # Citing papers (papers that cite this paper)
        citations: list[dict[str, Any]] = []
        for citation in paper_data.get("citations", [])[:max_citations]:
            citations.append({
                "title": citation.get("title", ""),
                "year": citation.get("year"),
                "authors": [
                    author.get("name", "")
                    for author in citation.get("authors", [])[:3]  # First 3 authors only
                ],
            })

        # References (papers that this paper cites)
        references: list[dict[str, Any]] = []
        for reference in paper_data.get("references", [])[:max_references]:
            references.append({
                "title": reference.get("title", ""),
                "year": reference.get("year"),
                "authors": [
                    author.get("name", "")
                    for author in reference.get("authors", [])[:3]
                ],
            })

        # Build analysis results
        paper_year = paper_data.get("year", 2025)
        citation_count = paper_data.get("citationCount", 0)
        years_since_publication = max(1, 2025 - paper_year) if paper_year else 1
        
        analysis = {
            "paper_id": paper_data.get("paperId"),
            "title": paper_data.get("title"),
            "year": paper_year,
            "citation_count": citation_count,
            "reference_count": paper_data.get("referenceCount", 0),
            "influential_citation_count": paper_data.get("influentialCitationCount", 0),
            "citations": citations,
            "references": references,
            "citation_velocity": citation_count / years_since_publication,
        }

        logger.info(
            f"Analyzed citations for: {analysis['title']} "
            f"(Citations: {analysis['citation_count']}, References: {analysis['reference_count']})"
        )
        return json.dumps(analysis, ensure_ascii=False, indent=2)

    except requests.exceptions.RequestException as e:
        error_msg = f"Error accessing Semantic Scholar API: {e!s}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)
    except Exception as e:
        error_msg = f"Error analyzing citations: {e!s}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

