"""
arXiv API integration for fetching paper information.
Extracts arXiv IDs from model tags and fetches paper metadata.
"""
import re
import httpx
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET


def extract_arxiv_ids(text: str) -> List[str]:
    """
    Extract arXiv IDs from text.
    Handles formats like:
    - arxiv:1234.5678
    - arxiv:1234.5678v1
    - https://arxiv.org/abs/1234.5678
    - arXiv:1234.5678
    """
    if not text:
        return []
    
    arxiv_ids = []
    
    # Pattern 1: arxiv:1234.5678 or arxiv:1234.5678v1
    pattern1 = r'arxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)'
    matches = re.findall(pattern1, text, re.IGNORECASE)
    arxiv_ids.extend(matches)
    
    # Pattern 2: https://arxiv.org/abs/1234.5678
    pattern2 = r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)'
    matches = re.findall(pattern2, text, re.IGNORECASE)
    arxiv_ids.extend(matches)
    
    # Pattern 3: arXiv:1234.5678 (with capital A)
    pattern3 = r'arXiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)'
    matches = re.findall(pattern3, text)
    arxiv_ids.extend(matches)
    
    # Remove version suffix for API calls (arXiv API works with base ID)
    arxiv_ids = [re.sub(r'v\d+$', '', id) for id in arxiv_ids]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for id in arxiv_ids:
        if id not in seen:
            seen.add(id)
            unique_ids.append(id)
    
    return unique_ids


async def fetch_arxiv_paper(arxiv_id: str) -> Optional[Dict]:
    """
    Fetch paper information from arXiv API.
    
    Args:
        arxiv_id: arXiv ID (e.g., "2103.00020" or "2508.06811")
    
    Returns:
        Dictionary with paper information or None if not found
    """
    try:
        # Remove version suffix if present
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # arXiv API endpoint
        url = f"http://export.arxiv.org/api/query"
        params = {
            "id_list": arxiv_id,
            "max_results": 1
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Namespace for arXiv API
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is None:
                return None
            
            # Extract paper information
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown Title"
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            
            # Extract published date
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text if published_elem is not None else None
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract arXiv ID from id field
            id_elem = entry.find('atom:id', ns)
            arxiv_url = id_elem.text if id_elem is not None else f"https://arxiv.org/abs/{arxiv_id}"
            
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "categories": categories,
                "url": arxiv_url
            }
            
    except Exception as e:
        # Error fetching arXiv paper
        return None


async def fetch_arxiv_papers(arxiv_ids: List[str]) -> List[Dict]:
    """
    Fetch multiple arXiv papers.
    
    Args:
        arxiv_ids: List of arXiv IDs
    
    Returns:
        List of paper dictionaries (None entries filtered out)
    """
    import asyncio
    
    # Fetch papers concurrently (limit to 5 at a time to avoid rate limits)
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_with_limit(arxiv_id: str):
        async with semaphore:
            return await fetch_arxiv_paper(arxiv_id)
    
    tasks = [fetch_with_limit(arxiv_id) for arxiv_id in arxiv_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None and exceptions
    papers = []
    for result in results:
        if result is not None and not isinstance(result, Exception):
            papers.append(result)
    
    return papers

