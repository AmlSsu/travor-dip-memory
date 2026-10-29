import unittest
from concurrent.futures import ThreadPoolExecutor

import test_contract as fixtures
from adaptive_evidence.core import Config, MemoryStore
from adaptive_evidence.retrieval_views import MODES, RetrievalViews


class ViewTests(unittest.TestCase):
    setUp = fixtures.ContractTests.setUp
    tearDown = fixtures.ContractTests.tearDown
    add = fixtures.ContractTests.add

    def test_porter_finds_inflection_and_query_syntax_is_data(self):
        self.add(content="Luna enjoys hiking through forests.")
        views = RetrievalViews(self.store.index("u"))
        self.assertTrue(views.lexical('hike OR "forest*"', "turns", 10))
        self.assertEqual(views.lexical('" * : -', "turns", 10), [])

    def test_all_modes_preserve_sources_scope_and_budget(self):
        self.add(content="Luna enjoys hiking.")
        self.add(user="other", content="Secret other-user destination.")
        for mode in MODES:
            result = self.store.search(user_id="u", query="Luna hiking", mode=mode)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], "[user] Luna enjoys hiking.")
            self.assertLessEqual(sum(len(r["content"]) for r in result), self.store.config.context_chars)

    def test_context_never_crosses_ordered_add_boundary(self):
        self.add(request="r1", content="Luna enjoys hiking.")
        self.add(request="r2", content="Alex builds violins.")
        index = self.store.index("u")
        views = RetrievalViews(index)
        for i in range(2):
            self.assertEqual(index.neighbors[i], [i])
        self.assertEqual(views.lexical("violins", "windows", 10), views.lexical("violins", "turns", 10))

    def test_add_invalidates_derived_views(self):
        self.add(request="r1", content="Luna enjoys hiking.")
        self.store.search(user_id="u", query="hiking", mode="context")
        old = self.store.index("u")
        self.add(request="r2", content="Luna bought a telescope.")
        self.assertIsNot(old, self.store.index("u"))
        result = self.store.search(user_id="u", query="telescope", mode="context", top_k=1)
        self.assertIn("telescope", result[0]["content"])

    def test_concurrent_readers_are_deterministic(self):
        self.add(content="Luna enjoys hiking through forests.")
        def search(_):
            return self.store.search(user_id="u", query="hike", mode="context_diverse")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(search, range(24)))
        self.assertTrue(all(r == results[0] for r in results))
