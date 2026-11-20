"""ChromaDB setup. One collection per company: physical tenant separation.

The collection name is always derived from the company in the verified JWT,
so even a forgotten metadata filter cannot return another company's chunks.

The rest of the app talks to Chroma only through the small methods below.
That keeps Chroma-specific code in one file, and lets tests substitute a
store that fails halfway through a write.
"""

import re
from functools import cached_property
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

_COMPANY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]$")


class UnpaddedMiniLM(ONNXMiniLM_L6_V2):
    """Chroma's default model (all-MiniLM-L6-v2, ONNX), without fixed-length padding.

    The stock version pads every input to 256 tokens, so a 10-token question
    costs as much as a full chunk (~175 ms on a laptop CPU). Padding tokens are
    masked out of the mean pooling, so dropping them gives the same vectors
    (checked by a test) at a fraction of the cost. Texts are encoded one at a
    time because unpadded texts have different lengths.

    Relies on the parent's `tokenizer` / `_forward` internals (chromadb pinned
    in requirements.txt). It keeps the parent's name(), so collections created
    with the default function stay compatible.
    """

    @cached_property
    def tokenizer(self):
        tok = super().tokenizer
        tok.no_padding()
        return tok

    def _forward(self, documents, batch_size: int = 32):
        return np.concatenate([super(UnpaddedMiniLM, self)._forward([d], batch_size=1) for d in documents])


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        # Local ONNX all-MiniLM-L6-v2: free, CPU-fast, no PyTorch.
        # The model is downloaded on first use (pre-fetched in the Docker image).
        self.embedding_function = UnpaddedMiniLM()

    def collection_for(self, company_id: str) -> Collection:
        if not _COMPANY_ID.match(company_id):
            raise ValueError(f"invalid company id: {company_id!r}")
        return self.client.get_or_create_collection(
            name=f"policy_{company_id}",
            configuration={"hnsw": {
                "space": "cosine",
                # Write the index to disk on every add. With the default (1000), a small
                # index lives only in Chroma's in-memory cache; if the cache evicts it
                # (capacity = file-handle limit / 5, i.e. ~100 indexes on Windows), the
                # collection fails with "Error creating hnsw segment reader: Nothing
                # found on disk". Policies are tiny, so the extra writes cost nothing.
                "sync_threshold": 1,
            }},
            embedding_function=self.embedding_function,
        )

    # --- writes (used by ingestion) --------------------------------------

    def add(self, company_id: str, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        self.collection_for(company_id).add(ids=ids, documents=documents, metadatas=metadatas)

    def count_version(self, company_id: str, version_id: int) -> int:
        found = self.collection_for(company_id).get(where={"version_id": version_id}, include=[])
        return len(found["ids"])

    def delete_version(self, company_id: str, version_id: int) -> None:
        self.collection_for(company_id).delete(where={"version_id": version_id})

    # --- reads (used by retrieval) ------------------------------------------------

    def query(self, company_id: str, text: str, n_results: int, version_ids: list[int]) -> list[dict]:
        """Nearest chunks in this company's collection, restricted to the given versions.

        Returns [{"id", "distance", "metadata"}], closest first. Distance is cosine
        distance (0 = identical direction).
        """
        res = self.collection_for(company_id).query(
            query_texts=[text],
            n_results=n_results,
            where={"version_id": {"$in": version_ids}},
            include=["metadatas", "distances"],
        )
        return [
            {"id": i, "distance": d, "metadata": m}
            for i, d, m in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
        ]

    # --- health ---------------------------------------------------------------

    def is_alive(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False

    def warm_up(self) -> None:
        """Run one embedding so the first real request does not pay model-load time."""
        self.embedding_function(["warm up"])
