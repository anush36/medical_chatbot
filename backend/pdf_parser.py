# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

import fitz  # PyMuPDF
import io
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using PyMuPDF.
    
    Args:
        file_bytes: Binary content of the PDF file.
        
    Returns:
        Extracted text string.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return f"[Error parsing PDF: {e}]"
