"""Load geological documents from disk.

Supports loading text (.txt) and PDF (.pdf) geological reports.
Preserves filename and basic metadata.
"""

from pathlib import Path
from typing import List, Dict, Any

from src.logger import get_logger
from src.pdf_processor import extract_pdf


class Document:
    """Represents a geological document."""

    def __init__(
        self,
        document_id: str,
        title: str,
        content: str,
        source_file: str,
        source_url: str = "",
        document_type: str = "geological_report",
        page_metadata: List[Dict[str, Any]] | None = None,
    ):
        self.document_id = document_id
        self.title = title
        self.content = content
        self.source_file = source_file
        self.source_url = source_url
        self.document_type = document_type
        self.page_metadata = page_metadata or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "source_file": self.source_file,
            "source_url": self.source_url,
            "document_type": self.document_type,
            "page_metadata": self.page_metadata,
        }


class DocumentLoader:
    """Load geological documents from a directory.

    Loads both .txt and .pdf files.

    When both a PDF and TXT version of the same report exist,
    the PDF is preferred and the duplicate TXT file is skipped.
    """

    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.logger = get_logger(__name__)

    def load_all(self) -> List[Document]:
        """Load all supported documents from the reports directory."""
        documents = []

        if not self.reports_dir.exists():
            self.logger.warning(
                f"Reports directory does not exist: {self.reports_dir}"
            )
            return documents

        # Find PDF and TXT files.
        pdf_files = list(self.reports_dir.glob("*.pdf"))
        txt_files = list(self.reports_dir.glob("*.txt"))

        # When a PDF and TXT have the same filename stem,
        # prefer the PDF and skip the TXT duplicate.
        pdf_stems = {file_path.stem for file_path in pdf_files}

        txt_files = [
            file_path
            for file_path in txt_files
            if file_path.stem not in pdf_stems
        ]

        files = sorted(pdf_files + txt_files)

        self.logger.info(
            f"Found {len(files)} supported files in {self.reports_dir}"
        )

        for file_path in files:
            try:
                doc = self.load_file(file_path)
                documents.append(doc)
                self.logger.debug(f"Loaded: {file_path.name}")
            except Exception as e:
                self.logger.error(
                    f"Error loading {file_path.name}: {e}"
                )

        self.logger.info(
            f"Successfully loaded {len(documents)} documents"
        )

        return documents

    def load_file(self, file_path: Path) -> Document:
        """Load a single TXT or PDF document."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return self._load_txt(file_path)

        if suffix == ".pdf":
            return self._load_pdf(file_path)

        raise ValueError(
            f"Unsupported file format: {file_path.suffix}"
        )

    def _load_txt(self, file_path: Path) -> Document:
        """Load a TXT document using the existing behavior."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        document_id = file_path.stem

        title = file_path.stem.replace("_", " ").replace("-", " ")

        first_line = content.split("\n")[0].strip()

        if first_line and len(first_line) < 200:
            title = first_line

        return Document(
            document_id=document_id,
            title=title,
            content=content,
            source_file=file_path.name,
            source_url="",
            document_type="geological_report",
        )

    def _load_pdf(self, file_path: Path) -> Document:
        """Load a PDF document using the PDF processor."""
        result = extract_pdf(file_path)

        page_metadata = result["pages"]

        content = "\n\n".join(
            page["text"]
            for page in page_metadata
            if page["text"].strip()
        )

        document_id = file_path.stem

        title = file_path.stem.replace("_", " ").replace("-", " ")

        first_non_empty_page = next(
            (
                page["text"].splitlines()[0].strip()
                for page in page_metadata
                if page["text"].strip()
                and page["text"].splitlines()
                and page["text"].splitlines()[0].strip()
            ),
            "",
        )

        if first_non_empty_page and len(first_non_empty_page) < 200:
            title = first_non_empty_page

        return Document(
            document_id=document_id,
            title=title,
            content=content,
            source_file=file_path.name,
            source_url="",
            document_type="geological_report",
            page_metadata=page_metadata,
        )
