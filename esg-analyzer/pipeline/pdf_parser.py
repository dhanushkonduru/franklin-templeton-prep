"""
PDF Parser for ESG documents.
Extracts text from PDF reports and performs cleaning/normalization.
"""

import fitz  # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import hashlib

from config import Config, setup_logging

logger = setup_logging(__name__)


class PDFParser:
    """Parse and extract text from PDF documents."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.config = Config()
        self.page_number_pattern = re.compile(r'^\s*[\d]+\s*$', re.MULTILINE)
        self.multiple_spaces_pattern = re.compile(r' {2,}')
        self.multiple_newlines_pattern = re.compile(r'\n{3,}')
    
    def parse_pdf(self, pdf_path: str | Path) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a PDF and extract text.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
            
        Raises:
            FileNotFoundError: If PDF not found
            ValueError: If PDF is corrupted or too large
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.MAX_PDF_SIZE_MB:
            raise ValueError(f"PDF too large: {file_size_mb:.2f}MB (max {self.config.MAX_PDF_SIZE_MB}MB)")
        
        try:
            logger.info(f"Parsing PDF: {pdf_path.name}")
            doc = fitz.open(pdf_path)
            
            text_parts = []
            page_texts = []
            
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                page_texts.append(page_text)
                text_parts.append(page_text)
            
            doc.close()
            
            raw_text = "\n\n".join(text_parts)
            cleaned_text = self._clean_text(raw_text)
            
            metadata = {
                "file_name": pdf_path.name,
                "file_size_mb": file_size_mb,
                "total_pages": len(page_texts),
                "raw_text_length": len(raw_text),
                "cleaned_text_length": len(cleaned_text),
                "file_hash": self._compute_file_hash(pdf_path),
                "extraction_successful": True,
            }
            
            logger.info(f"Successfully parsed {pdf_path.name}: {len(page_texts)} pages, {len(cleaned_text)} chars")
            return cleaned_text, metadata
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path.name}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove page numbers (lines with only digits)
        text = self.page_number_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.multiple_spaces_pattern.sub(' ', text)
        text = self.multiple_newlines_pattern.sub('\n\n', text)
        
        # Remove header/footer patterns (common in reports)
        text = self._remove_headers_footers(text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Attempt to remove common headers and footers.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with headers/footers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip common header/footer patterns
            if self._is_header_footer(stripped):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_header_footer(self, line: str) -> bool:
        """
        Detect if a line is likely a header or footer.
        
        Args:
            line: Line to check
            
        Returns:
            True if likely header/footer
        """
        if len(line) < 3:
            return False
        
        # Common patterns
        patterns = [
            r'^©\s*\d{4}',  # Copyright notice
            r'^page\s+\d+',  # Page number indicator
            r'^www\.',  # Website footer
            r'^confidential',  # Confidentiality notice
            r'\[page\s+\d+\]',  # Bracketed page number
        ]
        
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (uses config if None)
            overlap: Overlap between chunks (uses config if None)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.PDF_CHUNK_SIZE
        overlap = overlap or self.config.PDF_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        
        return chunks