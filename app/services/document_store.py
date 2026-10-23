"""Persistent SQLite store for document metadata and lifecycle states."""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import DocumentMetadata, DocumentProcessingStatus

logger = logging.getLogger("app.services.document_store")


class DocumentStore:
    """Manages document records and lifecycle persistence in SQLite."""

    def __init__(self, db_path: str | Path = "./data/research_assistant.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT,
                    file_size_bytes INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def save_document(
        self,
        metadata: DocumentMetadata,
        file_size_bytes: int = 0,
    ) -> None:
        """Insert or replace a document record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, filename, status, page_count, chunk_count,
                    created_at, updated_at, error_message, file_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.document_id,
                    metadata.filename,
                    metadata.status.value
                    if isinstance(metadata.status, DocumentProcessingStatus)
                    else str(metadata.status),
                    metadata.page_count,
                    metadata.chunk_count,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    metadata.error_message,
                    file_size_bytes,
                ),
            )
            conn.commit()
            logger.info(
                "Saved document %s (%s) with status '%s'",
                metadata.document_id,
                metadata.filename,
                metadata.status,
            )

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        """Retrieve metadata for a document by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_metadata(row)

    def list_documents(self, skip: int = 0, limit: int = 50) -> list[DocumentMetadata]:
        """List documents with pagination ordered by newest first."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, skip),
            )
            rows = cursor.fetchall()
            return [self._row_to_metadata(r) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document record by ID. Returns True if found and deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_status(
        self,
        document_id: str,
        status: DocumentProcessingStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update the processing status of a document."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE documents
                SET status = ?, updated_at = ?, error_message = ?
                WHERE document_id = ?
                """,
                (
                    status.value if isinstance(status, DocumentProcessingStatus) else str(status),
                    now,
                    error_message,
                    document_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """Return total number of registered documents."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

    def _row_to_metadata(self, row: sqlite3.Row) -> DocumentMetadata:
        created_at_dt = datetime.fromisoformat(row["created_at"])
        updated_at_dt = datetime.fromisoformat(row["updated_at"])
        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=UTC)
        if updated_at_dt.tzinfo is None:
            updated_at_dt = updated_at_dt.replace(tzinfo=UTC)

        return DocumentMetadata(
            document_id=row["document_id"],
            filename=row["filename"],
            status=DocumentProcessingStatus(row["status"]),
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            error_message=row["error_message"],
        )
