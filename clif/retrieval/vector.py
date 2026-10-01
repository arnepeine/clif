"""Embedders and the in-memory vector index.

Embedders (all on-prem):
- ``hashing``: deterministic character n-gram feature hashing. No model, no
  downloads. Captures spelling similarity and shared morphemes, not meaning.
- ``sentence-transformers``: a local multilingual model directory (e.g.
  multilingual-e5-small). Enables real cross-lingual semantic retrieval.
- ``http``: an OpenAI-compatible ``/v1/embeddings`` endpoint inside the
  hospital network (TEI, vLLM, Ollama ...).

The vector matrix is stored as a float32 .npy file and memory-mapped, so the
OS page cache keeps it resident. If faiss is installed it is used for search.
"""

from __future__ import annotations

import json
import logging
import threading
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np

from clif.config import Settings
from clif.text import norm_fold, tokens

log = logging.getLogger(__name__)


class Embedder(ABC):
    name: str = "embedder"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str], kind: str = "query") -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.dim}"


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _features(self, text: str) -> dict[int, float]:
        feats: dict[int, float] = {}
        for tok in tokens(norm_fold(text)):
            padded = f"#{tok}#"
            grams = [("w", tok, 1.0)]
            for n in (3, 4):
                grams.extend(("c", padded[i : i + n], 0.5) for i in range(max(1, len(padded) - n + 1)))
            for kind, g, w in grams:
                h = zlib.crc32(f"{kind}:{g}".encode())
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 else -1.0
                feats[idx] = feats.get(idx, 0.0) + sign * w
        return feats

    def embed(self, texts: list[str], kind: str = "query") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for idx, v in self._features(t).items():
                out[i, idx] = v
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 256):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model_name}:{self.dim}"

    def embed(self, texts: list[str], kind: str = "query") -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        vecs = self._model.encode(
            [prefix + t for t in texts], batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class HttpEmbedder(Embedder):
    name = "http"

    def __init__(self, url: str, model: str, dim: int = 0, query_prefix: str = "", passage_prefix: str = "",
                 timeout: float = 30.0):
        import httpx

        self._client = httpx.Client(timeout=timeout)
        self.url = url
        self.model_name = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.dim = dim or len(self.embed(["dimension probe"])[0])

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model_name}:{self.dim}"

    def embed(self, texts: list[str], kind: str = "query") -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        r = self._client.post(self.url, json={"model": self.model_name, "input": [prefix + t for t in texts]})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        vecs = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def make_embedder(settings: Settings) -> Embedder | None:
    if settings.embedder == "none":
        return None
    if settings.embedder == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    if settings.embedder == "sentence-transformers":
        if not settings.embedder_model:
            raise ValueError("CLIF_EMBEDDER_MODEL must point to a local sentence-transformers model")
        return SentenceTransformerEmbedder(settings.embedder_model, settings.embedder_query_prefix,
                                           settings.embedder_passage_prefix, settings.embed_batch_size)
    if settings.embedder == "http":
        return HttpEmbedder(settings.embedder_url, settings.embedder_model,
                            query_prefix=settings.embedder_query_prefix,
                            passage_prefix=settings.embedder_passage_prefix)
    raise ValueError(f"unknown embedder {settings.embedder}")


class CachedEmbedder:
    """Thread-safe LRU cache of query embeddings in front of an embedder."""

    def __init__(self, embedder: Embedder, size: int = 8192):
        self.embedder = embedder
        self.size = size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _get(self, text: str) -> np.ndarray | None:
        with self._lock:
            v = self._cache.get(text)
            if v is not None:
                self._cache.move_to_end(text)
            return v

    def _put(self, text: str, vec: np.ndarray) -> None:
        vec.setflags(write=False)
        with self._lock:
            self._cache[text] = vec
            self._cache.move_to_end(text)
            while len(self._cache) > self.size:
                self._cache.popitem(last=False)

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        """Embed all uncached texts in a single batched call."""
        missing = list(dict.fromkeys(t for t in texts if self._get(t) is None))
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing, kind="query")):
                self._put(t, np.array(v, dtype=np.float32))
        out = []
        for t in texts:
            v = self._get(t)
            if v is None:  # evicted between batch and read under heavy load
                v = self.embedder.embed([t], kind="query")[0]
            out.append(v)
        return out

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]


class VectorIndex:
    def __init__(self, directory: Path):
        meta = json.loads((directory / "meta.json").read_text())
        self.identity: str = meta["embedder"]
        self.dim: int = meta["dim"]
        self.matrix = np.load(directory / "embeddings.npy", mmap_mode="r")
        self.row_ids = np.load(directory / "row_ids.npy")
        self._faiss = None
        try:
            import faiss  # optional

            idx = faiss.IndexFlatIP(self.dim)
            idx.add(np.ascontiguousarray(self.matrix, dtype=np.float32))
            self._faiss = idx
            log.info("Vector index: faiss IndexFlatIP with %d vectors", len(self.row_ids))
        except ImportError:
            log.info("Vector index: numpy brute force with %d vectors", len(self.row_ids))

    def __len__(self) -> int:
        return len(self.row_ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        n = len(self.row_ids)
        if n == 0:
            return []
        k = min(k, n)
        if self._faiss is not None:
            scores, idx = self._faiss.search(query.reshape(1, -1).astype(np.float32), k)
            return [(int(self.row_ids[i]), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        best_idx: list[np.ndarray] = []
        best_scores: list[np.ndarray] = []
        chunk = 200_000
        for start in range(0, n, chunk):
            s = np.asarray(self.matrix[start : start + chunk]) @ query
            kk = min(k, len(s))
            part = np.argpartition(-s, kk - 1)[:kk]
            best_idx.append(part + start)
            best_scores.append(s[part])
        idx = np.concatenate(best_idx)
        sc = np.concatenate(best_scores)
        order = np.argsort(-sc)[:k]
        return [(int(self.row_ids[idx[i]]), float(sc[i])) for i in order]


def write_vector_index(directory: Path, identity: str, dim: int, row_ids: np.ndarray, matrix: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", matrix.astype(np.float32))
    np.save(directory / "row_ids.npy", row_ids.astype(np.int64))
    (directory / "meta.json").write_text(json.dumps({"embedder": identity, "dim": dim, "count": int(len(row_ids))}))
