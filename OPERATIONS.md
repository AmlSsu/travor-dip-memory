# Context Weave / SourceTrail 0.32.0

Raw-source Add and deterministic Search; zero generative LLM calls in Add/Search. Final Answer belongs to the external evaluator. Original author: 王子铭 (Giorno-Banana), MIT; official display Adaptive Evidence.

This release retains 0.31's query-local document detection and ranking. When at least three of the twelve leading dense chunks come from one original user document of at least 2,000 characters, the default query_document_evidence mode protects directly ranked evidence before expanding large source windows or literal same-source section references. Other conversations retain 0.31 ranking and packing. Every role remains stored and retrievable, returned bodies are contiguous original spans, and date metadata is preserved. The experimental raw-BM25 stopword filter was rejected and is absent from this release.

Embeddings remain text-embedding-v4, 1024 dimensions, batch 10. Search uses the same single query/option embedding call, 24,000-character context budget and top_k <= 100. No generated summaries, query expansion, LLM planner, reranker, or judge. Authentication, atomic writes, request_id idempotency, immediate retrieval and user isolation remain.

## Reproduce

Install requirements.txt, copy env.example to a private environment file, set credentials and a separate AE_DATABASE, then run:

```powershell
.\start_local.ps1 -Port 18098 -PythonPath C:\service-venv\Scripts\python.exe -EnvFile @('.env')
python preflight.py --env-file .env --live --output synthetic-probe.json
python -m unittest discover -s tests
```

## Evidence and limits

64 contract and retrieval tests pass, including protection of a second direct source from large first-source expansion, exact source fidelity, budgets, roles, timestamps, isolation and unchanged query embedding calls. All 40 reused public BEAM contexts are identical to 0.31, requiring no new final Answer generation. Public LoCoMo complete-evidence counts remain 137/152 and 137/153. These panels were previously inspected and are regression evidence, not unseen validation.

Public CL-Bench retrieval changes four development and six confirmation contexts. With the same local Qwen3.5-4B/runtime0.4.2 final Answer and exact-prompt answer reuse, there is no established complete-answer gain: the correct development answer is retained, several failures remain, and confirmation has a new local Answer timeout plus one preserved prior timeout. Six new final Answer calls were made across both panels, with no judge API. Codex reviewed public semantics, not an independent human. The predeclared strict improvement gate is not met; official Smoke is exploratory. Neither public diagnostics nor a single Smoke prove Full performance or repeatability.

This archive contains implementation, tests and public configuration only: no credentials, databases, weights, datasets or benchmark answers. The versioned archive is authoritative; historical root implementation is retained for provenance. See PROVENANCE.md for primary method sources and excluded generative methods.
