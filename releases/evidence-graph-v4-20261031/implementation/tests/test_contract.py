import concurrent.futures
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from adaptive_evidence.api import app, get_store
from adaptive_evidence.core import Config, ConflictError, MemoryStore, split_spans, tokens


class FakeEncoder:
    """Deterministic lexical embedding for contract tests, never a measured model."""
    identity = "test-only-hash-256"
    fail = False

    def encode(self, texts, *, query=False):
        if self.fail:
            raise RuntimeError("encoder unavailable")
        values = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokens(text):
                values[i, int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 256] += 1
            if not values[i].any():
                values[i, 0] = 1
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "memory.db"
        self.encoder = FakeEncoder()
        self.store = MemoryStore(self.path, self.encoder)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.directory.cleanup()

    def add(self, user="u", request="r", content="Luna likes the blue feather wand.", **kwargs):
        return self.store.add(user_id=user, request_id=request, session_id="session",
                              messages=[{"role": "user", "content": content, **kwargs}])

    def test_immediate_search_isolated_by_user_not_session(self):
        self.add(user="alice")
        self.add(user="bob", content="Luna likes a red ball.")
        alice = self.store.search(user_id="alice", query="Luna toy")
        bob = self.store.search(user_id="bob", query="Luna toy")
        self.assertIn("blue feather", alice[0]["content"])
        self.assertNotIn("red ball", str(alice))
        self.assertIn("red ball", bob[0]["content"])
        self.assertEqual([], self.store.search(user_id="unknown", query="Luna"))
        self.assertNotEqual(alice[0]["id"], bob[0]["id"])

    def test_identical_retry_and_conflicting_retry(self):
        self.assertGreater(self.add(), 0)
        self.assertEqual(0, self.add())
        with self.assertRaises(ConflictError):
            self.add(content="Different memory")
        self.assertEqual(1, len(self.store.search(user_id="u", query="Luna")))

    def test_concurrent_add_idempotence(self):
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _: self.add(), range(16)))
        self.assertEqual(1, len(self.store.search(user_id="u", query="Luna")))

    def test_failure_has_no_partial_write(self):
        self.encoder.fail = True
        with self.assertRaises(RuntimeError):
            self.add()
        self.encoder.fail = False
        self.assertEqual([], self.store.search(user_id="u", query="Luna"))
        self.assertGreater(self.add(), 0)

    def test_cache_invalidates_after_add_in_another_process_store(self):
        self.add()
        self.store.search(user_id="u", query="Luna")
        second = MemoryStore(self.path, FakeEncoder())
        second.add(user_id="u", request_id="r2", session_id="other",
                   messages=[{"role": "user", "content": "The updated preference is a red ball."}])
        hits = self.store.search(user_id="u", query="updated red ball")
        self.assertEqual(2, len(hits))
        self.assertIn("red ball", hits[0]["content"])

    def test_source_offsets_cover_long_input_and_keep_tail(self):
        text = ("The first sentence. \n" * 100) + "最后一条信息🙂 FINALSECRET"
        spans = split_spans(text, 200, 30)
        covered = set()
        for start, end in spans:
            covered.update(range(start, end))
        self.assertEqual(set(range(len(text))), covered)
        self.store = MemoryStore(Path(self.directory.name) / "small.db", self.encoder,
                                 Config(chunk_chars=200, overlap_chars=30))
        self.add(content=text)
        hit = self.store.search(user_id="u", query="FINALSECRET", top_k=1)[0]
        self.assertIn("FINALSECRET", hit["content"])
        self.assertIn(hit["content"].split("] ", 1)[1], text)

    def test_timestamp_missing_not_replaced_by_ingestion_date(self):
        self.add()
        hit = self.store.search(user_id="u", query="Luna", top_k=1)[0]
        self.assertNotIn("created_at", hit)
        self.assertTrue(hit["content"].startswith("[user] "))

    def test_old_and_new_versions_preserve_source_dates(self):
        self.add(request="old", content="My home is Paris.", timestamp=1704067200000)
        self.add(request="new", content="My home is Berlin now.", timestamp=1735689600000)
        hits = self.store.search(user_id="u", query="Where is my home?")
        self.assertEqual(2, len(hits))
        self.assertEqual({"2024", "2025"}, {h["created_at"][:4] for h in hits})

    def test_neighbors_do_not_cross_add_or_session_boundaries(self):
        self.add(request="a")
        self.add(request="b", content="Unrelated secret")
        index = self.store.index("u")
        self.assertTrue(all(neighbors == [i] for i, neighbors in index.neighbors.items()))

    def test_top_k_budget_and_scores(self):
        self.store = MemoryStore(Path(self.directory.name) / "budget.db", self.encoder,
                                 Config(context_chars=200))
        for i in range(10):
            self.add(request=str(i), content=f"Luna preference number {i}: blue wand.")
        hits = self.store.search(user_id="u", query="Luna", top_k=3)
        self.assertLessEqual(len(hits), 3)
        self.assertLessEqual(sum(len(h["content"]) for h in hits), 200)
        self.assertEqual(sorted([h["score"] for h in hits], reverse=True), [h["score"] for h in hits])
        with self.assertRaises(ValueError):
            self.store.search(user_id="u", query="Luna", top_k=101)

    def test_model_identity_rejects_stale_database(self):
        self.add()
        encoder = FakeEncoder()
        encoder.identity = "different-model"
        with self.assertRaises(ValueError):
            MemoryStore(self.path, encoder)

    def test_textual_roles_preserved_and_empty_messages_rejected(self):
        self.store.add(user_id="u", request_id="tool-r", session_id="s",
                       messages=[{"role": "tool", "content": "Retrieved source: Paris"}])
        self.assertTrue(self.store.search(user_id="u", query="Paris")[0]["content"].startswith("[tool]"))
        for content in ("", "   "):
            with self.assertRaises(ValueError):
                self.add(request="empty", content=content)

    def test_purge_clears_persistent_data_and_cache(self):
        self.add()
        self.store.search(user_id="u", query="Luna")
        self.store.purge_user("u")
        self.assertEqual([], self.store.search(user_id="u", query="Luna"))
        self.assertEqual([], MemoryStore(self.path, self.encoder).search(user_id="u", query="Luna"))

    def test_api_contract_auth_and_no_label_inputs(self):
        app.dependency_overrides[get_store] = lambda: self.store
        client = TestClient(app)
        payload = {"user_id": "u", "session_id": "s", "request_id": "r",
                   "messages": [{"role": "user", "content": "Luna blue wand"}]}
        with patch.dict(os.environ, {"MEMORY_API_KEY": "local-test-only", "AE_LOCAL_NO_AUTH": "0"}):
            self.assertEqual(200, client.get("/health").status_code)
            self.assertEqual(401, client.post("/add", json=payload).status_code)
            auth = {"Authorization": "Bearer local-test-only"}
            reply = client.post("/add", json=payload, headers=auth)
            self.assertEqual({"success": True, "user_id": "u", "session_id": "s", "request_id": "r"}, reply.json())
            search = {"user_id": "u", "query": "Luna toy", "top_k": 1}
            reply = client.post("/search", json=search, headers=auth)
            self.assertEqual(1, len(reply.json()["data"]))
            forbidden = {**search, "gold_answer": "private-test-marker"}
            reply = client.post("/search", json=forbidden, headers=auth)
            self.assertEqual(422, reply.status_code)
            self.assertNotIn("private-test-marker", reply.text)


if __name__ == "__main__":
    unittest.main()
