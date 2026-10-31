FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY adaptive_evidence ./adaptive_evidence
RUN useradd --uid 10001 --create-home memory && mkdir -p /data && chown memory:memory /data
ENV AE_DATABASE=/data/context-weave-v280/memory.sqlite3 AE_EMBEDDING_BACKEND=dashscope PYTHONDONTWRITEBYTECODE=1
USER memory
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "adaptive_evidence.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
