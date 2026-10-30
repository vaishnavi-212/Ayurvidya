"""Vector tests: the embedding cache, the Gemini embedder (network mocked) and ChromaDB (skipped if not installed)."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ayurvidya.embed import CachedEmbedder, GeminiEmbedder  # noqa: E402
from ayurvidya.schema import Chunk  # noqa: E402


class FakeDense:
    name = "fake"

    def __init__(self):
        self.seen = 0

    def fit(self, texts):
        pass

    def embed_documents(self, texts):
        self.seen += len(texts)
        return np.array([[len(t), 1.0, 0.5] for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return np.array([len(text), 1.0, 0.5], dtype=np.float32)

    def save(self, path):
        pass


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cache_embeds_each_text_once(self):
        inner = FakeDense()
        e = CachedEmbedder(inner, self.tmp / "v.npz")
        e.embed_documents(["a", "bb", "ccc"])
        e2 = CachedEmbedder(FakeDense(), self.tmp / "v.npz")       # a later rebuild
        m = e2.embed_documents(["a", "bb", "dddd"])
        self.assertEqual(e2.inner.seen, 1)                          # only the new text
        self.assertEqual(m.shape, (3, 3))

    def test_gemini_request_shape(self):
        class R:
            status_code = 200

            def json(self):
                return {"embeddings": [{"values": [3.0, 4.0]}, {"values": [0.0, 2.0]}]}
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "k"}), mock.patch("requests.post", return_value=R()) as p:
            v = GeminiEmbedder(dim=2).embed_documents(["प्राणः", "उदानः"])
        body = p.call_args.kwargs["json"]
        self.assertEqual(body["requests"][0]["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(body["requests"][0]["outputDimensionality"], 2)
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=5)

    def test_chroma_store_filters_by_type(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("chromadb not installed")
        from ayurvidya.index import ChromaVectorStore
        chunks = [Chunk("a", "x", "mula"), Chunk("b", "y", "commentary"), Chunk("c", "z", "mula")]
        m = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        st = ChromaVectorStore([c.id for c in chunks], m, self.tmp / "chroma", chunks, sha="s1")
        hits = st.search(np.array([1, 0], dtype=np.float32), 2, source_types=["mula"])
        self.assertEqual([i for i, _ in hits], [0, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)