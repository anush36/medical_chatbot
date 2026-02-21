import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Base URLs for NCBI E-utilities and BioC API
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOC_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"

def get_pmcids_from_query(query: str, max_results: int = 15) -> List[str]:
    """Search PMC using E-utilities and return a list of PMCIDs."""
    search_url = f"{EUTILS_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pmc",
        # Force the search to only return articles that have full text in the Open Access subset
        "term": f"({query}) AND open access[filter]",
        "retmode": "xml",
        "retmax": max_results,
        "sort": "relevance"
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        pmcids = []
        for id_elem in root.findall(".//Id"):
            pmcid = id_elem.text
            # Prefix with PMC if not already, BioC API expects PMCxxxxx format
            if pmcid and not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            pmcids.append(pmcid)
            
        logger.info(f"Retrieved PMCIDs for query '{query}': {pmcids}")
        return pmcids
    except Exception as e:
        logger.error(f"Error fetching PMCIDs from E-utilities: {e}")
        return []

def get_bioc_content(pmcid: str) -> str:
    """Fetch the BioC JSON content for a specific PMCID and extract text."""
    url = f"{BIOC_BASE_URL}/{pmcid}/unicode"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # If it returns HTML instead of JSON (e.g. not open access), this will fail cleanly
        data = response.json()
        
        # Handle BioC JSON structure which typically returns a BioCCollection
        # A BioCCollection contains a list of BioCDocuments
        collections = data if isinstance(data, list) else [data]
        
        extracted_text = []
        target_sections = {"TITLE", "ABSTRACT", "INTRO", "RESULTS", "CONCLUSIONS", "DISCUSSION"}
            
        for collection in collections:
            if not isinstance(collection, dict):
                continue
                
            docs = collection.get("documents", [collection])
            for doc in docs:
                for passage in doc.get("passages", []):
                    section_type = passage.get("infons", {}).get("section_type", "").upper()
                    if section_type in target_sections or not section_type:
                        if "text" in passage and passage["text"]:
                            extracted_text.append(passage["text"])
                        
        full_text = "\n".join(extracted_text)
        
        max_chars = 6000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "... [TRUNCATED]"
            
        return full_text
        
    except ValueError as e: # JSONDecodeError inherits from ValueError
        logger.warning(f"Invalid JSON received for {pmcid}, possibly not in open access subset: {e}")
        return ""
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not fetch BioC data for {pmcid}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error parsing BioC data for {pmcid}: {e}")
        return ""

@tool
def search_pmc(query: str) -> str:
    """
    Search PubMed Central (PMC) for medical literature.
    Provides snippets of text from relevant research articles to ground medical claims.
    
    Args:
        query: The search terms, e.g., 'acute myeloid leukemia treatments'
    Returns:
        Summarized text extracts from the top retrieved articles.
    """
    logger.info(f"Tool call: search_pmc with query '{query}'")
    
    pmcids = get_pmcids_from_query(query, max_results=15)
    if not pmcids:
        return f"No articles found for query: {query}"
        
    combined_texts = []
    for pmcid in pmcids:
        text = get_bioc_content(pmcid)
        if text:
            abstract_snippet = f"--- Source: {pmcid} ---\n{text}\n"
            combined_texts.append(abstract_snippet)
            if len(combined_texts) >= 3:
                break
            
    if not combined_texts:
        return f"Found articles for '{query}' but could not extract readable text."
        
    return "\n".join(combined_texts)
