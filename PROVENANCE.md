# 来源与方法边界

作者与维护者：王子铭（[Giorno-Banana](https://github.com/Giorno-Banana)）。本项目新代码采用 [MIT 许可证](LICENSE)。

2026-09-29：0.2.0 默认 Embedding 改为百炼 `text-embedding-v4`（1024 维）；BGE 仅保留在历史实验路径。适配器依据[百炼原生文本向量同步 API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api/)独立实现，未包含其模型权重。方法与验证范围见 README、SUBMISSION 和 PILOT_RESULTS。

本目录的实现依据本项目历史实验和公开方法思想重新编写，没有复制四个上游仓库的源文件或提示词。历史复现代码仍位于项目原来的 `repos/` 和 `benchmarks/`，与本目录分开。

方法来源应随正式提交披露：

- [InvMem / vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory)：原始记忆、稠密/词法混合检索与 RRF 的参照。
- [ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex)：原文优先和相邻原文窗口的参照；本候选尚未迁移 AMI 事实抽取。
- [ReFind](https://github.com/imlrz/ReFind)：按问题搜集多处证据的参照；本候选规划模块是有界的两阶段实现，不是 ReFind 原四步状态机。
- [Mem0](https://github.com/mem0ai/mem0)：记忆更新、实体与历史证据保留的研究参照；本候选不包含 Mem0 SDK 或其托管优化。
- 本项目 `benchmarks/hybrid_combinations`：Adaptive-All4 的自适应检索及上下文预算实验。
- 2026-09-07 小样本研究新增 [FlowGrid](https://github.com/dlxeva/flowgrid-aml-retriever) 的来源关联多视图、[aml-memory-mvp](https://github.com/0xboyu/aml-memory-mvp) 的互补证据选择，以及 [LongMemEval](https://arxiv.org/html/2410.10813v2) 的多键索引思路参照；新代码为独立实现，没有复制其源文件。SQLite FTS5 Porter 为标准库运行时能力。上下文向量由现有 BGE 向量池化，未复现论文的事实抽取或 HippoRAG 2 的知识图谱。
- 历史研究工作区另存了前十仓库版本、文件 SHA 与论文链接；发布包不包含该历史实验记录。公开主分支不一定等于榜单评分版本。
- [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)：历史实验使用的嵌入模型；发布模型时需保留模型自己的许可证与模型卡。0.2.0 拟提交配置使用 text-embedding-v4。
- [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined)：本地公开诊断数据，发布候选包不附带数据或答案。

API 契约依据 [AML API Guide](https://agentmemories.ai/api-guide)。历史规划模块接口依据 [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) 和 [GPT-4o-mini](https://developers.openai.com/api/docs/models/gpt-4o-mini)。第二期规则已于 2026-09-29 重新核验，当前按赛事 FAQ 的模型要求准备，详见 README_CYCLE2.md。

本项目的 MIT 许可仅覆盖新代码，不改变上述项目、第三方依赖、模型和数据的许可。发布包不包含上游仓库、模型权重、原始诊断数据或标准答案。

## 0.3.0 Add 阶段更新（2026-10-06）

新增通过 [OpenRouter Chat Completions](https://openrouter.ai/docs/quickstart) 调用 `openai/gpt-4o-mini` 的适配器。新增的 Add 索引处理使用模型选择原文短语，代码验证其为原文子串，再用于向量与 BM25 索引加权；未复制第三方实现。模型输出不会作为新事实或最终答案返回。提供方固定为 OpenAI，使用[结构化输出](https://openrouter.ai/docs/guides/features/structured-outputs)及[提供方参数约束](https://openrouter.ai/docs/guides/routing/provider-selection)。这是新的方法配置，其效果需要单独验证；历史基线分数不能直接移作本版本成绩。

## 0.3.1 reliability update (2026-10-08)

The same authors replaced free-form cue copying with integer selection of deterministic source spans, added private per-request batch checkpoints and sanitized diagnostic logs. No third-party implementation was copied. This changes the method identity and requires an independently bound version and a fresh database. It does not claim a measured quality improvement.

## 0.4.0 local model option (2026-10-08)

Added an explicitly selected loopback Chat Completions backend, actual-model reporting and model/runtime profile isolation. Shared source-span validation and the original OpenRouter GPT-4o-mini request behavior are retained. Local compatibility adds the schema to the prompt and has its own identity.

Local validation used existing [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) weights; its local model card declares Apache-2.0. Model weights, tokenizer and third-party library binaries are not redistributed in this package. local_runtime/ contains the exact existing project model-server source and a runtime/weight hash manifest; the model remains subject to its own license. This profile is not covered by the 0.3.1 official Smoke result.

## 0.4.1 runtime reliability update (2026-10-08)

Added a supervised GPU worker process, bounded request queue, cancellation/deadlines, shared single/batch tensor cleanup and sanitized runtime metrics. These are original project changes. The original 0.4.0 loading helpers remain, while supervised serving uses the new generation lifecycle. All nine external model/tokenizer file hashes were rechecked and match 0.4.0. Source-span prompts, indexing/retrieval behavior and cloud fallback policy are unchanged. The runtime identity and database are new; the 0.4.0 official Smoke score does not certify this version.


## 0.6.0 local evidence planner

Original implementation of bounded query planning and source-identifier selection. Method inspiration: ReFind, revision a80175ca0eeb52a938d7cab7a602bc780de8a577, app/retriever.py (https://github.com/imlrz/ReFind). No upstream source/prompt was copied. Also informed by this project's earlier public select6 experiments; their local scores are not official scores. Add and embedding behavior remain based on 0.4.1.

## 0.6.1 batch-selection correction

Original project change based on public diagnostic failures: explicit irrelevant-batch abstention, a validated 16-ID limit, and deterministic BM25 accumulation. No additional upstream code or prompt was copied.

## 0.27.0 raw-source memory (2026-10-13)

Original project change: remove generative Add indexing and ReFind-inspired Search planning from this release. Retain original-source hybrid retrieval and source windows. Generative Answer belongs to the external evaluator. Embedding remains text-embedding-v4. Historical sections above describe prior releases, not enabled components in this package. No upstream code or prompt was added.

## 0.28.0 deterministic evidence windows (2026-10-13)

Original implementation: budget-aware neighbor reservation, exact overlap merging and within-Add source ordering, combined with the existing SQLite Porter lexical view. No new model, third-party code or prompts. Add and embedding settings are unchanged from 0.27.0. Public paired retrieval diagnostics are described in README.md; no official score is implied.


##0.29.0 relevant source admission (2026-10-13)

Original implementation of relevance-first source packing, bounded distinct-source admission, exact source overlap merging and bounded original user standing-instruction reservation. Existing embeddings and query calls are unchanged; production uses focus_stem_rules. Research-only contextual, fine-passage and phrase variants are not selected for this release.

Method inspiration, with independent code: MemFusion's single-pass dense/BM25 reciprocal-rank fusion and bounded neighbor expansion (https://github.com/huangpengtao00-dotcom/memfusion-memory/blob/main/wiki_store.py; verified blob5e7c138f85b2ecd38c6c24bee2e84fd904520ebf); MemMachine's best-match-first episode admission and bounded context folding (https://github.com/MemMachine/MemMachine/blob/main/packages/server/src/memmachine_server/episodic_memory/long_term_memory/long_term_memory.py; verified blob544bc828b2fcf2f3adffd48caf1dbd58e73fea88). Their generative Add profiling, wiki writing, query expansion, agent loops and model reranking are excluded.

The public comparison holds stored sources and embedding identity constant, informed by What Should an Agent Remember? Disentangling Retention from Retrieval (https://arxiv.org/abs/2610.00366). Remember by Asking: RIME (https://arxiv.org/abs/2609.34438) informed selective original dialogue context; its LLM fact consolidation and Answer fallback are excluded. This release does not claim to reproduce either full paper. No upstream code/prompt was copied.

## 0.30.0 conditional original-source provenance (2026-10-14)

Original project implementation: add an original-user Porter lexical ranking stream only for scopes dominated by one long original user document; otherwise preserve 0.29 ranking. No generative extraction, model reranking, or extra embedding request. The MemFusion and MemMachine sources above informed existing lexical fusion and bounded source admission. Compile, Don't Memorize (https://github.com/TonyQJH/cca-emnlp2026) informed investigation of rule completeness and source provenance; its LLM compilation, code generation, and answer correction are explicitly excluded. This is not a reproduction of CCA.

Public CL-Bench data (https://huggingface.co/datasets/tencent/CL-bench) was used only in isolated local diagnostics. The primary authors' repository mirror was verified against the publisher's SHA-256. No benchmark-specific answer, task identifier, or public/private dataset is included in the source package. Dataset licenses remain separate from the implementation's MIT license.
## 0.31.0 query-local source coherence (2026-10-14)

Independent generic routing using already-computed dense scores, original role/offsets, and literal same-source section references. Direct document lookup and prior-answer references use distinct lexical views while preserving all source roles. No upstream code or prompt copied. MemFusion/MemMachine sources and retention-versus-retrieval/RIME papers above informed bounded original context; their extra model calls are excluded. CL-Bench (https://arxiv.org/abs/2602.03587) and Compile, Don't Memorize (https://github.com/TonyQJH/cca-emnlp2026) informed public investigation of source completeness. CCA LLM compilation, code generation and answer correction are excluded; this is not a reproduction. No task identifiers or benchmark answers occur in production routing.

## 0.32.0 direct evidence budget (2026-10-14)

Independently implemented source packing protects direct matches before bounded original source expansion. Ranking, prompt, embeddings and call counts are unchanged from 0.31. Public tests are isolated from official evaluation. No task IDs, reference answers or external code copied into production. This adapts original-source coherence and bounded expansion described above; no generative consolidation or compiler is introduced.
