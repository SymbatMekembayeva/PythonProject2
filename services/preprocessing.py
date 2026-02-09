from pathlib import Path
from typing import List
import PyPDF2


def pdf2chunks(filepath: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Extract text from a PDF file and split it into overlapping chunks.

    Args:
        filepath: Path to the PDF file
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks as strings
    """
    # Extract all text from PDF
    full_text = ""
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            full_text += page.extract_text() + " "

    # Split into chunks with overlap
    chunks = []
    start = 0

    while start < len(full_text):
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = full_text[start:end]

        chunks.append(chunk)

        # Move start forward by (chunk_size - chunk_overlap)
        start += chunk_size - chunk_overlap

        # Break if we've reached the end
        if end >= len(full_text):
            break

    return chunks