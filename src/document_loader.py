"""Load geological documents from disk.

Supports loading text documents (.txt files) from a directory.
Preserves filename and basic metadata.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from src.logger import get_logger


class Document:
    """Represents a geological document.
    
    Attributes:
        document_id: Unique identifier (based on filename).
        title: Document title (derived from filename or content).
        content: Full text content.
        source_file: Original filename.
        source_url: Optional URL (for scraped documents).
        document_type: Type of document (e.g., 'geological_report').
    """
    
    def __init__(
        self,
        document_id: str,
        title: str,
        content: str,
        source_file: str,
        source_url: str = "",
        document_type: str = "geological_report"
    ):
        self.document_id = document_id
        self.title = title
        self.content = content
        self.source_file = source_file
        self.source_url = source_url
        self.document_type = document_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "source_file": self.source_file,
            "source_url": self.source_url,
            "document_type": self.document_type,
        }


class DocumentLoader:
    """Load geological documents from a directory.
    
    Loads all .txt files from a directory, extracting basic metadata
    from filenames and content.
    """
    
    def __init__(self, reports_dir: Path):
        """Initialize document loader.
        
        Args:
            reports_dir: Path to directory containing .txt files.
        """
        self.reports_dir = Path(reports_dir)
        self.logger = get_logger(__name__)
    
    def load_all(self) -> List[Document]:
        """Load all .txt files from reports directory.
        
        Returns:
            List of Document objects.
        """
        documents = []
        
        if not self.reports_dir.exists():
            self.logger.warning(
                f"Reports directory does not exist: {self.reports_dir}"
            )
            return documents
        
        txt_files = list(self.reports_dir.glob("*.txt"))
        self.logger.info(f"Found {len(txt_files)} text files in {self.reports_dir}")
        
        for file_path in txt_files:
            try:
                doc = self.load_file(file_path)
                documents.append(doc)
                self.logger.debug(f"Loaded: {file_path.name}")
            except Exception as e:
                self.logger.error(
                    f"Error loading {file_path.name}: {e}"
                )
        
        self.logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def load_file(self, file_path: Path) -> Document:
        """Load a single .txt file.
        
        Args:
            file_path: Path to .txt file.
        
        Returns:
            Document object.
        
        Raises:
            ValueError: If file is not .txt or doesn't exist.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != ".txt":
            raise ValueError(f"File must be .txt format: {file_path}")
        
        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 for compatibility
            content = file_path.read_text(encoding="latin-1")
        
        # Generate document ID from filename (without extension)
        document_id = file_path.stem
        
        # Extract title: either from filename or first line of content
        title = file_path.stem.replace("_", " ").replace("-", " ")
        first_line = content.split("\n")[0].strip()
        if first_line and len(first_line) < 200:  # Reasonable title length
            title = first_line
        
        return Document(
            document_id=document_id,
            title=title,
            content=content,
            source_file=file_path.name,
            source_url="",
            document_type="geological_report"
        )
