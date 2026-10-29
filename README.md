# travor‘s dip — Textual Memory 占位基线

本仓库用于余付（GitHub: [AmlSsu](https://github.com/AmlSsu)）以个人身份参加
Agent Memory Challenge 的文本记忆 / 开源方法赛道，供赛事方检查当前部署源码。
**目前为临时占位版本，后续将更新参赛代码。**

当前实现完整复用 Context Weave / SourceTrail **0.32.0**，未修改算法、API、模型配置
或测试。原作者为王子铭（[Giorno-Banana](https://github.com/Giorno-Banana)），
采用 [MIT 许可证](LICENSE)。本仓库不把原作者的历史分数作为本队成绩，也不声称
此占位基线是本队的新方法。

## 供赛事方核验

- 独立部署版本标记：`0.32.0-placeholder`；未修改的 API `/health` 报告 `0.32.0`。
- 原始公开源码提交：[`e84af580a3a9a21f1eab6328d41e8b93bb52552e`](https://github.com/Giorno-Banana/context-weave/tree/e84af580a3a9a21f1eab6328d41e8b93bb52552e)。
- 权威源码包：[context-weave-0.32.0-source.zip](context-weave-0.32.0-source.zip)。
- 源码包 SHA256：`bff4443ee3cbdf4a5e320dfa56fb8e65ca7775d857550c4112de7ddb7a9c74fa`。
- [BASELINE_IDENTITY.json](BASELINE_IDENTITY.json) 列出原始文件与仓库路径、逐文件 SHA256，
  并说明本次仅有文档与发布包装变更。
- 原作者的说明与历史验证范围保存在 [UPSTREAM_README_0.32.0.md](UPSTREAM_README_0.32.0.md)；
  其他原始文档中的作者与实验描述也属于上游历史记录。
- [PROVENANCE.md](PROVENANCE.md) 保留来源、相关工作及许可证边界。

服务代码位于 `adaptive_evidence/`，原始测试位于 `tests/`。Add 存储原文，Search 返回
原始证据；Add/Search 不调用生成式 LLM。当前采用 `text-embedding-v4`、1024 维、
batch 10、`query_document_evidence` 与 24000 字符本地证据预算。Answer 和 Eval
由平台统一执行。

## 当前独立部署

| 项目 | 地址 / 说明 |
| --- | --- |
| Add | `https://wzm.tail36b9f2.ts.net:8443/add` |
| Search | `https://wzm.tail36b9f2.ts.net:8443/search` |
| Health | `https://wzm.tail36b9f2.ts.net:8443/health` |
| 鉴权 | `X-Api-Key`；密钥仅通过赛事申请表的专用字段提供 |
| 本机端口 | `127.0.0.1:18140` |
| 隔离 | 独立源码副本、进程、端口、SQLite 数据库与服务密钥 |
| 共享基础设施 | 主机、Python 运行时、Tailscale 域名及上游 Embedding 账户 |
| 初始联调并发 | Add 1 / Search 1 |

本机服务使用独立数据库，不转发到其他参赛接口，也不共享评测记忆。个人电脑需保持
接电、联网且不休眠；重启后需重新启动服务。已完成合成数据的健康、鉴权、Add/Search、
用户隔离及重启持久化检查；尚未完成独立远程主机的容量验收，也未为本队运行官方评测。

## 复现

安装 Python 3.10+，在本仓库根目录创建环境并安装锁定依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item env.example .env
```

在私有 `.env` 中填写 `MEMORY_API_KEY`、`DASHSCOPE_API_KEY` 和真实
`AE_EMBEDDING_URL`，并将 `AE_DATABASE` 设置为本队独立路径，例如
`data/travor-dip-placeholder/memory.sqlite3`。保留其余 0.32.0 配置；不要提交此文件。

```powershell
.\start_local.ps1 -Port 18140 -PythonPath .\.venv\Scripts\python.exe -EnvFile @('.env')
```

在另一个终端运行原始离线契约与检索测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

`preflight.py --live` 会调用真实 Embedding 并产生相应用量，按需单独执行；它不是
官方 Smoke 或 Full。部署时应使用独立数据库和专用服务密钥。

## 后续版本

后续修改会通过新 commit 披露方法差异、来源和对应部署版本。正式评测前需向赛事方
固定对应的公开 commit 与 API 版本。本仓库不包含真实密钥、数据库、模型权重、数据集
或比赛答案。此次发布不改变正在运行的占位服务代码。
