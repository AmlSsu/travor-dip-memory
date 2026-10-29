import unittest

import test_contract as fixtures
from adaptive_evidence.core import Config, MemoryStore


class AdaptiveTests(unittest.TestCase):
    setUp = fixtures.ContractTests.setUp
    tearDown = fixtures.ContractTests.tearDown
    add = fixtures.ContractTests.add

    def test_enumeration_uses_direct_hybrid_order(self):
        for i in range(10):
            self.add(request=str(i), content=f"Luna visited city number {i}.")
        args = dict(user_id="u", query="List all cities Luna visited", top_k=5)
        self.assertEqual(self.store.search(**args, mode="hybrid"), self.store.search(**args, mode="adaptive"))

    def test_small_budget_keeps_neighbors_searchable(self):
        self.store = MemoryStore(self.path.parent / "short.db", self.encoder, Config(context_chars=350))
        self.store.add(user_id="u", request_id="r", session_id="s", messages=[
            {"role": "user", "content": "Luna bought the violin."},
            {"role": "assistant", "content": "It happened yesterday, immediately after the concert."},
            {"role": "user", "content": "Luna practiced playing the violin."},
            {"role": "user", "content": "Luna talked about the violin."},
        ])
        hits = self.store.search(user_id="u", query="When did Luna buy the violin?", mode="adaptive")
        self.assertIn("yesterday", str(hits))
        self.assertLessEqual(sum(len(h["content"]) for h in hits), 350)
