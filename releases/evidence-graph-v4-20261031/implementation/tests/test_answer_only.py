import json
import os
import socket
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

import preflight
from adaptive_evidence import api
from adaptive_evidence.core import MemoryStore
from adaptive_evidence.remote_encoder import DashScopeEncoder
from test_contract import FakeEncoder
from test_remote_encoder import ENDPOINT, result


class AnswerOnlyTests(unittest.TestCase):
    def tearDown(self):
        api.app.dependency_overrides.clear()
        api.get_store.cache_clear()

    def settings(self, database):
        # Deliberately inherited old flags must not restore any LLM code path.
        return {"MEMORY_API_KEY": "synthetic-service-key", "DASHSCOPE_API_KEY": "synthetic-embedding-key",
                "AE_EMBEDDING_URL": ENDPOINT, "AE_DATABASE": str(database),
                "AE_ADD_INDEXER": "1", "AE_PLANNER": "1", "AE_LOCAL_EVIDENCE_SEARCH": "1",
                "AE_LOCAL_EVIDENCE_EXPAND": "1", "AE_LOCAL_LLM_URL": "http://127.0.0.1:1/v1/chat/completions"}

    def test_add_search_lifespan_never_contact_chat_despite_inherited_flags(self):
        calls = []
        def embedding_only(request):
            self.assertEqual(str(request.url), ENDPOINT)
            body = json.loads(request.content)
            self.assertNotIn("messages", body)
            self.assertEqual(body["model"], "text-embedding-v4")
            calls.append(body)
            return httpx.Response(200, json=result(len(body["input"]["texts"])))
        encoder = DashScopeEncoder("synthetic-embedding-key", ENDPOINT, dimensions=64,
                                   transport=httpx.MockTransport(embedding_only))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self.settings(Path(tmp)/"new.db"), clear=True), \
             patch.object(api, "encoder_from_env", return_value=encoder), \
             patch.object(socket, "create_connection", side_effect=AssertionError("Unexpected network/chat call")):
            with TestClient(api.app) as client:
                health = client.get("/health").json()
                self.assertEqual(health["version"], "0.32.0")
                self.assertEqual(health["add_llm"], "none")
                self.assertEqual(health["search_planner"], "none")
                auth = {"X-Api-Key": "synthetic-service-key"}
                body = {"user_id":"alice", "request_id":"r", "session_id":"s",
                        "messages":[{"role":"user", "content":f"Original memory number {i}: blue train."} for i in range(23)]}
                self.assertEqual(client.post("/add",headers=auth,json=body).status_code,200)
                self.assertEqual([len(x["input"]["texts"]) for x in calls], [10,10,3])
                self.assertEqual(calls[0]["input"]["texts"][0],body["messages"][0]["content"])
                count=len(calls)
                self.assertEqual(client.post("/add",headers=auth,json=body).status_code,200)
                self.assertEqual(len(calls),count)
                response=client.post("/search",headers=auth,json={"user_id":"alice","query":"blue train","top_k":3})
                self.assertEqual(response.status_code,200)
                self.assertTrue(response.json()["data"])
                self.assertTrue(all(x["content"].startswith("[user] Original memory") for x in response.json()["data"]))
                self.assertEqual(calls[-1]["parameters"]["text_type"],"query")
                self.assertEqual(client.post("/search",headers=auth,json={"user_id":"bob","query":"blue train","top_k":3}).json(),{"data":[]})
            self.assertTrue(encoder.client.is_closed)

    def test_core_rejects_an_llm_indexer(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexer=Mock()
            with self.assertRaisesRegex(ValueError,"raw-source"):
                MemoryStore(Path(tmp)/"synthetic.db",FakeEncoder(),add_indexer=indexer)
            indexer.enrich.assert_not_called()

    def test_index_text_and_sources_are_unmodified_in_synthetic_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/"synthetic.db",FakeEncoder())
            source="Do not replace original evidence. "+("Long source with tail. "*100)+"最后🙂"
            store.add(user_id="a",request_id="r",session_id="s",messages=[{"role":"user","content":source}])
            with closing(store.connect()) as db:
                self.assertEqual(db.execute("SELECT content FROM sources").fetchone()[0],source)
                for row in db.execute("SELECT start,end,index_text FROM chunks"):
                    self.assertEqual(row[2],source[row[0]:row[1]])

    def test_preflight_no_llm_credentials_or_network_required(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,self.settings(Path(tmp)/"new.db"),clear=True), \
             patch.object(httpx.Client,"post",side_effect=AssertionError("offline")):
            report=preflight.configuration_report()
            self.assertTrue(report["ready_for_live_probe"])
            self.assertEqual(report["add_llm"],"none")
            self.assertNotIn("synthetic-embedding-key",json.dumps(report))
            self.assertNotIn("synthetic-service-key",json.dumps(report))

    def test_synthetic_preflight_uses_only_embedding_provider(self):
        calls=[]
        def handler(request):
            self.assertEqual(str(request.url),ENDPOINT)
            body=json.loads(request.content);calls.append(body)
            return httpx.Response(200,json=result(len(body["input"]["texts"])))
        encoder=DashScopeEncoder("synthetic",ENDPOINT,dimensions=64,transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,self.settings(Path(tmp)/"new.db"),clear=True), \
             patch.object(api,"encoder_from_env",return_value=encoder):
            report=preflight.live_probe()
        self.assertEqual(report["live_probe"],"passed")
        self.assertIsNone(report["add_llm_usage"])
        self.assertGreater(len(calls),5)


if __name__ == "__main__":
    unittest.main()
