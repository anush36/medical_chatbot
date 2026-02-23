# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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

def get_bioc_content(pmcid: str) -> Dict[str, str]:
    """Fetch the BioC JSON content for a specific PMCID and extract title and text."""
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
        title_text = "Unknown Title"
        target_sections = {"TITLE", "ABSTRACT", "INTRO", "RESULTS", "CONCLUSIONS", "DISCUSSION"}
            
        for collection in collections:
            if not isinstance(collection, dict):
                continue
                
            docs = collection.get("documents", [collection])
            for doc in docs:
                for passage in doc.get("passages", []):
                    section_type = passage.get("infons", {}).get("section_type", "").upper()
                    if section_type == "TITLE" and "text" in passage:
                        title_text = passage["text"]
                    if section_type in target_sections or not section_type:
                        if "text" in passage and passage["text"]:
                            extracted_text.append(passage["text"])
                        
        full_text = "\n".join(extracted_text)
        
        # Generous truncation since we are chunking it anyway 
        max_chars = 25000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "... [TRUNCATED]"
            
        return {"title": title_text, "text": full_text}
        
    except ValueError as e: # JSONDecodeError inherits from ValueError
        logger.warning(f"Invalid JSON received for {pmcid}, possibly not in open access subset: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not fetch BioC data for {pmcid}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing BioC data for {pmcid}: {e}")
        return {}

@tool
def search_pmc(query: str) -> str:
    """
    Search PubMed Central (PMC) for medical literature.
    Provides highly relevant snippets of text from retrieved research articles to ground medical claims.
    
    Args:
        query: The search terms, e.g., 'acute myeloid leukemia treatments'
    Returns:
        Summarized text extracts from the top retrieved articles based on semantic similarity.
    """
    logger.info(f"Tool call: search_pmc with query '{query}'")
    
    # We fetch fewer full articles (top 4) to keep API calls fast, but extract deeper into them
    pmcids = get_pmcids_from_query(query, max_results=4)
    if not pmcids:
        return f"No articles found for query: {query}"
        
    docs = []
    
    # 1. Fetch full text and create Document objects
    for pmcid in pmcids:
        content = get_bioc_content(pmcid)
        if content and content.get("text"):
            title = content.get("title", "Unknown Title")
            docs.append(Document(
                page_content=content["text"],
                metadata={"source": pmcid, "title": title}
            ))
            
    if not docs:
        return f"Found articles for '{query}' but could not extract readable text."

    # 2. Semantic Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} articles into {len(chunks)} chunks.")

    # 3. Ephemeral Vector Search
    # Using a fast, lightweight local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create temporary in-memory chroma store
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    # Retrieve top k most relevant chunks (e.g. 5 chunks * 800 chars = ~4000 chars total)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    top_chunks = retriever.invoke(query)
    
    # 4. Format Output
    combined_texts = []
    for chunk in top_chunks:
        pmcid = chunk.metadata.get("source", "Unknown")
        title = chunk.metadata.get("title", "Unknown")
        snippet = f"--- Source: {pmcid} (Title: {title}) ---\n{chunk.page_content}\n"
        combined_texts.append(snippet)
        
    return "\n".join(combined_texts)
