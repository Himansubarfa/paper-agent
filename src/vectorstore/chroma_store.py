"""Chroma vector store integration for paper embeddings and retrieval."""

import re
from typing import Any, Optional
import chromadb
from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.llm.provider import embedder_slug
from src.logging_setup import get_logger
from src.models.schemas import Chunk

logger = get_logger(__name__)


class PaperStore:
    """Manages document chunk indexing, retrieval with MMR, and persistence in Chroma."""

    def __init__(self, doc_id: str, settings: Settings, embeddings: Embeddings) -> None:
        self.doc_id = doc_id
        self.settings = settings
        self.embeddings = embeddings
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))

    @property
    def collection_name(self) -> str:
        """Construct a sanitized collection name adhering to Chroma rules (3-63 alnum/-/_)."""
        slug = embedder_slug(self.settings)
        raw_name = f"{self.doc_id}_{slug}"
        # Sanitize to valid Chroma characters: alnum, _, -
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)[:63].strip("_-")
        if len(clean_name) < 3:
            clean_name = clean_name.ljust(3, "0")
        return clean_name

    def is_indexed(self) -> bool:
        """Check whether the collection exists and contains indexed items."""
        try:
            col = self._client.get_collection(name=self.collection_name)
            return col.count() > 0
        except Exception:
            return False

    def index(self, chunks: list[Chunk]) -> None:
        """Index a list of chunks into Chroma. Idempotent across multiple calls."""
        if not chunks:
            return

        col = self._client.get_or_create_collection(name=self.collection_name)
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "chunk_id": c.id,
                "page": c.page,
                "section_id": c.section_id,
                "section_heading": c.section_heading,
            }
            for c in chunks
        ]
        embeddings = self.embeddings.embed_documents(documents)
        col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        logger.info("Indexed %d chunks into collection '%s'", len(chunks), self.collection_name)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        section_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Retrieve relevant chunks using Maximal Marginal Relevance (MMR).

        Args:
            query: The search query.
            k: Number of chunks to return (defaults to retrieval_top_k).
            section_id: Optional section_id to filter results.

        Returns:
            List of matching Chunk objects.
        """
        if not self.is_indexed():
            return []

        top_k = k or self.settings.retrieval_top_k
        fetch_k = self.settings.retrieval_fetch_k

        from langchain_chroma import Chroma
        vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

        search_kwargs: dict[str, Any] = {
            "k": top_k,
            "fetch_k": fetch_k,
        }
        if section_id:
            search_kwargs["filter"] = {"section_id": section_id}

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

        docs = retriever.invoke(query)
        chunks: list[Chunk] = []
        for doc in docs:
            meta = doc.metadata
            chunks.append(
                Chunk(
                    id=meta.get("chunk_id", ""),
                    text=doc.page_content,
                    page=int(meta.get("page", 1)),
                    section_id=meta.get("section_id", ""),
                    section_heading=meta.get("section_heading", ""),
                )
            )
        return chunks

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch a specific chunk by its ID."""
        try:
            col = self._client.get_collection(name=self.collection_name)
            res = col.get(ids=[chunk_id], include=["documents", "metadatas"])
            if not res["ids"] or len(res["ids"]) == 0:
                return None
            text = res["documents"][0]
            meta = res["metadatas"][0]
            return Chunk(
                id=meta.get("chunk_id", chunk_id),
                text=text,
                page=int(meta.get("page", 1)),
                section_id=meta.get("section_id", ""),
                section_heading=meta.get("section_heading", ""),
            )
        except Exception as exc:
            logger.debug("Chunk %s not found in collection %s: %s", chunk_id, self.collection_name, exc)
            return None

    def delete(self) -> None:
        """Drop the underlying Chroma collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info("Deleted collection '%s'", self.collection_name)
        except Exception as exc:
            logger.warning("Could not delete collection '%s': %s", self.collection_name, exc)
