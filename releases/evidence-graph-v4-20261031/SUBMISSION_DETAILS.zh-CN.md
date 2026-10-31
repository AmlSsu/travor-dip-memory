# 冻结V4申报说明

系统：travor‘s dip。提交者：余付（GitHub AmlSsu）。机构/团队：DataSpot。

本目录是 exact SourceTrail / Context Weave 0.32.0 的派生实现，新增直接原文块图与联合选材。原作者王子铭及MIT许可完整保留；新贡献和来源见ATTRIBUTION.md及原始PROVENANCE.md。不是将原版改名再提交。

当前账号版本ID version_2b209886a6d2、原注册端点 :8443/add 和 :8443/search。沿用该已注册版本ID与显示名，实际方法明确标识为 evidence-graph-v4-20261031，以避免重新绑定和版本混淆；源码API包版本0.32.0保留为上游兼容标记，实际mode和源码指纹识别本候选。所有鉴权和部署私有参数均不公开。

首个匹配Smoke teval_e9b7392e81fbc239：2026-10-31T11:18:10Z成功，46/46有效，65.8854166667。该分数不能证明稳定优于incumbent；完整本地最终答案评分/永久性能门槛未通过。用户已明确允许准备这份可能不及基线的探索性Full；这不意味着实际POST已批准，也不意味着质量门槛通过或晋升。

Textual，top_k100，Full拟Add并发1/Search并发1（Smoke为16/16）。向量text-embedding-v4/1024维；Add/Search无生成LLM。Search仅返回同user持久化原文证据，平台统一完成Answer/Eval。不开启多模态或代码赛道，不改变官方提示、数据、模型和评分。

五项API合同已完成本地合成检查；C盘实际部署另通过61项公网合成合同检查；embedding预算已获不设上限授权。现有C盘约47.9GiB、可用RAM约2.25GiB，合成分批/随机向量场景外推12亿字符约7.28至44.34GiB；这不是官方Full预测或上界，最紧情形余量约3.5GiB。正式POST前工程准入下限为C盘46.5GiB空闲和1GiB可用RAM，并发1/1，仍须人类接受薄余量与持续在线安排。首次Full受理后固定版本且停止Smoke；不得因低分换版或撤回。评测数据仅用于当前任务，终态30天内删除自有数据及备份。公开仓库不含任何评测题目、原始记忆、答案、凭据或模型权重。

后续第二个同源Smoke teval_a5af4f60685b6b3d已46/46完成、59.0798611111，两次均值62.4826388889；不只报告较高的一次。SQLite默认4KiB页面保留，64KiB容量假设未节省空间，未进入生产。最终POST须在原注册端点上复验实际主服务原文合同与指纹，严格v2只提交version_id/benchmark_type/phase/options/run_label。
