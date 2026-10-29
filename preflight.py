"""Cycle 2 configuration check; --live calls only embeddings on synthetic data.

Default mode is offline. Neither mode submits an AML Smoke or Full task.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from adaptive_evidence.core import Config, MemoryStore
from adaptive_evidence.remote_encoder import EmbeddingError, encoder_from_env, validate_endpoint
from adaptive_evidence import __version__


def load_env(path: Path):
    """Read simple KEY=value lines as data, never execute or expand values."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            raise ValueError("Invalid environment file syntax")
        name, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[name] = value


def configuration_report():
    required = ["MEMORY_API_KEY", "DASHSCOPE_API_KEY", "AE_EMBEDDING_URL", "AE_DATABASE"]
    missing = [name for name in required if not os.getenv(name, "").strip()]
    problems = []
    if os.getenv("AE_EMBEDDING_URL"):
        try:
            validate_endpoint(os.environ["AE_EMBEDDING_URL"])
        except ValueError:
            problems.append("AE_EMBEDDING_URL is invalid or still contains a placeholder")
    if os.getenv("AE_EMBEDDING_BACKEND", "dashscope") != "dashscope":
        problems.append("This production profile uses AE_EMBEDDING_BACKEND=dashscope")
    if os.getenv("AE_LOCAL_NO_AUTH") == "1":
        problems.append("Disable AE_LOCAL_NO_AUTH for the submitted service")
    try:
        Config(context_chars=int(os.getenv("AE_CONTEXT_CHARS", "24000")))
    except ValueError:
        problems.append("Invalid AE_CONTEXT_CHARS")
    if os.getenv("AE_MODE", "query_document_evidence") not in {"hybrid_window", "hybrid", "adaptive", "dense", "evidence_window", "stem_window", "document_focus", "query_document_evidence"}:
        problems.append("Use a supported candidate retrieval mode")
    if not missing and not problems:
        encoder = None
        try:
            encoder = encoder_from_env()
        except ValueError:
            problems.append("Invalid embedding configuration")
        finally:
            if encoder is not None:
                encoder.close()
    return {"track": "textual", "version": __version__, "add_llm": "none",
            "search_llm": "none", "answer_owner": "external_evaluator",
            "embedding": "text-embedding-v4", "missing_environment": missing,
            "configuration_problems": problems, "ready_for_live_probe": not missing and not problems,
            "live_probe": "not_run", "official_smoke": False, "official_full": False}


def live_probe():
    from fastapi.testclient import TestClient
    from adaptive_evidence.api import app, get_store

    def check(condition, stage):
        if not condition:
            raise RuntimeError("Live probe failed at " + stage)

    with tempfile.TemporaryDirectory(prefix="ae-cycle2-probe-") as tmp:
        path = Path(tmp) / "synthetic.sqlite3"
        with patch.dict(os.environ, {"AE_DATABASE": str(path)}):
            get_store.cache_clear()
            with TestClient(app) as client:
                headers = {"X-Api-Key": os.environ["MEMORY_API_KEY"]}
                check(client.get("/health").status_code == 200, "health")
                body = {"request_id": "r1", "user_id": "synthetic-a", "session_id": "s1",
                        "messages": [{"role": "user", "timestamp": 1735689600000,
                                      "content": "My home city is Paris. 我的旅行偏好是乘火车。"}]}
                check(client.post("/add", json=body).status_code == 401, "authentication")
                response = client.post("/add", headers=headers, json=body)
                check(response.status_code == 200, "add")
                check(response.json() == {"success": True, "request_id": "r1", "user_id": "synthetic-a", "session_id": "s1"}, "add IDs")
                store = get_store()
                before = store.encoder.usage()["requests"]
                before_llm = store.add_indexer.usage()["requests"] if store.add_indexer else 0
                check(client.post("/add", headers=headers, json=body).status_code == 200, "idempotent retry")
                check(store.encoder.usage()["requests"] == before, "no duplicate embedding")
                if store.add_indexer:
                    check(store.add_indexer.usage()["requests"] == before_llm, "no duplicate LLM indexing")
                conflict = {**body, "messages": [{"role": "user", "content": "conflicting retry"}]}
                check(client.post("/add", headers=headers, json=conflict).status_code == 409, "conflict")
                query = {"user_id": "synthetic-a", "query": "我喜欢什么旅行方式？", "top_k": 100,
                         "options": ["A. 火车", "B. 飞机"]}
                response = client.post("/search", headers=headers, json=query)
                check(response.status_code == 200, "search")
                check("乘火车" in str(response.json()["data"]), "source evidence")
                other = {**body, "user_id": "synthetic-b", "messages": [{"role": "user", "content": "My home city is Tokyo."}]}
                check(client.post("/add", headers=headers, json=other).status_code == 200, "second user")
                response = client.post("/search", headers=headers, json={**query, "user_id": "synthetic-b"})
                check(response.status_code == 200 and "Tokyo" in str(response.json()) and "Paris" not in str(response.json()), "scope isolation")
                update = {**body, "request_id": "r2", "session_id": "s2",
                          "messages": [{"role": "user", "timestamp": 1767225600000, "content": "I moved to Berlin."}]}
                check(client.post("/add", headers=headers, json=update).status_code == 200, "streaming update")
                response = client.post("/search", headers=headers, json={**query, "query": "Where did I move?"})
                check(response.status_code == 200 and "Berlin" in str(response.json()), "cache invalidation")
                restored = MemoryStore(path, store.encoder, store.config, add_indexer=store.add_indexer)
                check(len(restored.search(user_id="synthetic-a", query="home city")) == 2, "persistent recovery")
                return {"live_probe": "passed", "transport": "in-process HTTP with configured upstream providers",
                        "synthetic_data_only": True, "embedding_usage": store.encoder.usage(),
                        "add_llm_usage": store.add_indexer.usage() if store.add_indexer else None,
                        "checks": ["health", "auth", "add_ids", "idempotence", "conflict", "retrieval",
                                   "choice_options", "user_isolation", "streaming_update", "persistence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, action="append",
                        help="Repeat for profile overlays; later files take precedence")
    parser.add_argument("--live", action="store_true", help="Spend only embedding calls on a small synthetic probe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.env_file:
        for path in args.env_file:
            load_env(path)
    report = configuration_report()
    if args.live and report["ready_for_live_probe"]:
        try:
            report.update(live_probe())
        except (EmbeddingError, RuntimeError, ValueError, OSError):
            report.update(live_probe="failed", error="Live probe failed; verify provider access and service configuration")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_live_probe"] and (not args.live or report["live_probe"] == "passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
