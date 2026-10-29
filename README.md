# travor‘s dip — Textual Memory 

本仓库用于余付（GitHub: [AmlSsu](https://github.com/AmlSsu)）以个人身份参加
Agent Memory Challenge 的文本记忆 / 开源方法赛道，供赛事方检查当前部署源码。

## 供赛事方核验

- 独立部署版本标记：`0.32.0-placeholder`；未修改的 API `/health` 报告 `0.32.0`。
- 源码包 SHA256：`bff4443ee3cbdf4a5e320dfa56fb8e65ca7775d857550c4112de7ddb7a9c74fa`。

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
