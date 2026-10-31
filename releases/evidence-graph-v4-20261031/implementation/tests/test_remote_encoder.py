import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

from adaptive_evidence.api import app, get_store
from adaptive_evidence.core import MemoryStore
from adaptive_evidence.remote_encoder import DashScopeEncoder, EmbeddingError, encoder_from_env


ENDPOINT = "https://embedding.example.test/api/v1/services/embeddings/text-embedding/text-embedding"


def result(count, dimension=64):
    # Deliberately reversed response order; vectors identify the original inputs.
    return {"output": {"embeddings": [
        {"text_index": i, "embedding": [1.0 if j == i % dimension else 0.0 for j in range(dimension)]}
        for i in reversed(range(count))]}, "usage": {"total_tokens": count * 2}}


class RemoteEncoderTests(unittest.TestCase):
    def encoder(self, handler, **kwargs):
        encoder = DashScopeEncoder("test-secret", ENDPOINT, dimensions=64,
                                   transport=httpx.MockTransport(handler), sleep=lambda _: None, **kwargs)
        self.addCleanup(encoder.close)
        return encoder

    def test_batch_roles_order_normalization_and_usage(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual("Bearer test-secret", request.headers["Authorization"])
            return httpx.Response(200, json=result(len(body["input"]["texts"])))

        encoder = self.encoder(handler)
        vectors = encoder.encode([f"memory {i}" for i in range(23)])
        self.assertEqual([10, 10, 3], [len(r["input"]["texts"]) for r in requests])
        self.assertEqual((23, 64), vectors.shape)
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1))
        self.assertEqual(list(range(10)), vectors[:10].argmax(axis=1).tolist())
        self.assertTrue(all(r["model"] == "text-embedding-v4" for r in requests))
        self.assertTrue(all(r["parameters"]["text_type"] == "document" for r in requests))
        encoder.encode(["query without BGE prefix"], query=True)
        self.assertEqual("query", requests[-1]["parameters"]["text_type"])
        self.assertEqual(["query without BGE prefix"], requests[-1]["input"]["texts"])
        self.assertEqual(48, encoder.usage()["reported_total_tokens"])
        self.assertEqual((0, 64), encoder.encode([]).shape)
        self.assertEqual(4, encoder.usage()["requests"])

    def test_transient_errors_retry_but_auth_and_redirect_do_not(self):
        statuses = iter([429, 503, 200])
        encoder = self.encoder(lambda _: httpx.Response(next(statuses), json=result(1)))
        encoder.encode(["memory"])
        self.assertEqual(3, encoder.usage()["requests"])
        for status in (400, 401, 403, 302):
            with self.subTest(status=status):
                failed = self.encoder(lambda _: httpx.Response(status, text="test-secret private-memory"))
                with self.assertRaises(EmbeddingError) as ctx:
                    failed.encode(["private-memory"])
                self.assertNotIn("test-secret", str(ctx.exception))
                self.assertNotIn("private-memory", str(ctx.exception))
                self.assertEqual(1, failed.usage()["requests"])

    def test_network_failure_bounded_and_private(self):
        def handler(request):
            raise httpx.ReadTimeout("private-memory test-secret", request=request)
        encoder = self.encoder(handler, max_attempts=2)
        with self.assertRaises(EmbeddingError) as ctx:
            encoder.encode(["private-memory"])
        self.assertEqual(2, encoder.usage()["requests"])
        self.assertNotIn("private-memory", str(ctx.exception))

    def test_malformed_vectors_never_accepted(self):
        bad_bodies = [
            {}, {"output": {"embeddings": []}},
            {"output": {"embeddings": [{"text_index": 0, "embedding": [0.] * 64}]}},
            {"output": {"embeddings": [{"text_index": 0, "embedding": [1.] * 63}]}},
            {"output": {"embeddings": [{"text_index": 1, "embedding": [1.] * 64}]}},
            {"output": {"embeddings": [{"text_index": False, "embedding": [1.] * 64}]}},
            {"output": {"embeddings": [{"text_index": 0, "embedding": [True] * 64}]}},
            {"output": {"embeddings": [{"text_index": 0, "embedding": ["1"] * 64}]}},
            {"code": "InvalidApiKey", "message": "test-secret"},
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                encoder = self.encoder(lambda _: httpx.Response(200, json=body))
                with self.assertRaises(EmbeddingError):
                    encoder.encode(["memory"])
                self.assertEqual(1, encoder.usage()["requests"])
        duplicate = result(2)
        duplicate["output"]["embeddings"][0]["text_index"] = 0
        encoder = self.encoder(lambda _: httpx.Response(200, json=duplicate))
        with self.assertRaises(EmbeddingError):
            encoder.encode(["one", "two"])
        encoder = self.encoder(lambda _: httpx.Response(200, content=b'{"output":{"embeddings":[{"text_index":0,"embedding":[NaN]}]}}'))
        with self.assertRaises(EmbeddingError):
            encoder.encode(["memory"])

    def test_request_budget_includes_retries_and_all_batches(self):
        encoder = self.encoder(lambda _: httpx.Response(429), max_requests=1)
        with self.assertRaisesRegex(EmbeddingError, "budget"):
            encoder.encode(["memory"])
        self.assertEqual(1, encoder.usage()["requests"])

    def test_identity_excludes_key_but_tracks_vector_space(self):
        encoder = self.encoder(lambda _: httpx.Response(200))
        rotated = DashScopeEncoder("rotated-secret", ENDPOINT, dimensions=64)
        changed = DashScopeEncoder("rotated-secret", ENDPOINT, dimensions=128)
        self.addCleanup(rotated.close)
        self.addCleanup(changed.close)
        self.assertEqual(encoder.identity, rotated.identity)
        self.assertNotEqual(encoder.identity, changed.identity)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            MemoryStore(path, encoder)
            with self.assertRaisesRegex(ValueError, "different model"):
                MemoryStore(path, changed)

    def test_missing_config_fails_without_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
                encoder_from_env()
        for endpoint in ("http://example.test/", "https://name:secret@example.test/", "https://example.test/?key=secret",
                         "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1"):
            with self.assertRaises(ValueError):
                DashScopeEncoder("test", endpoint)

    def test_http_failure_rolls_back_retry_persists_and_restart_reads(self):
        failed = False
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            if failed:
                return httpx.Response(503, text="private-upstream-payload")
            return httpx.Response(200, json=result(len(json.loads(request.content)["input"]["texts"])))

        encoder = self.encoder(handler, max_attempts=1, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            store = MemoryStore(path, encoder)
            app.dependency_overrides[get_store] = lambda: store
            self.addCleanup(app.dependency_overrides.clear)
            client = TestClient(app)
            with patch.dict(os.environ, {"MEMORY_API_KEY": "local-test", "AE_PLANNER": "0"}):
                headers = {"X-Api-Key": "local-test"}
                payload = {"request_id": "r", "session_id": "s", "user_id": "u",
                           "messages": [{"role": "user", "content": "I live in Paris."}]}
                failed = True
                reply = client.post("/add", headers=headers, json=payload)
                self.assertEqual(503, reply.status_code)
                self.assertNotIn("private-upstream-payload", reply.text)
                self.assertIsNone(store.index("u"))
                failed = False
                self.assertEqual(200, client.post("/add", headers=headers, json=payload).status_code)
                before = calls
                self.assertEqual(200, client.post("/add", headers=headers, json=payload).status_code)
                self.assertEqual(before, calls)
                restarted = MemoryStore(path, encoder)
                self.assertIn("Paris", restarted.search(user_id="u", query="home")[0]["content"])
                self.assertEqual([], restarted.search(user_id="another-user", query="home"))
                updated = {**payload, "request_id": "r2", "messages": [{"role": "user", "content": "I moved to Berlin."}]}
                self.assertEqual(200, client.post("/add", headers=headers, json=updated).status_code)
                self.assertEqual(2, len(restarted.search(user_id="u", query="home")))
                failed = True
                reply = client.post("/search", headers=headers, json={"user_id": "u", "query": "home", "top_k": 100})
                self.assertEqual(503, reply.status_code)

    def test_later_batch_failure_leaves_no_partial_memory(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(503) if calls == 2 else httpx.Response(200, json=result(1))
        encoder = self.encoder(handler, batch_size=1, max_attempts=1)
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db", encoder)
            with self.assertRaises(EmbeddingError):
                store.add(user_id="u", request_id="r", session_id="s",
                          messages=[{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}])
            self.assertIsNone(store.index("u"))


if __name__ == "__main__":
    unittest.main()
