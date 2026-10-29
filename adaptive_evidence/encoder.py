"""Local BGE encoder. Model loading is explicit and does not download by default."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np


class BGEEncoder:
    def __init__(self, path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        model_dir = Path(path).resolve()
        if not model_dir.is_dir():
            raise ValueError("Provide an existing local BGE model directory")
        # Hash actual weights and tokenizer, so a changed model cannot reuse old vectors.
        digest = hashlib.sha256()
        for name in ("config.json", "model.safetensors", "tokenizer.json", "vocab.txt"):
            file = model_dir / name
            if file.exists():
                digest.update(name.encode())
                with file.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        self.identity = "bge-cls-normalized-512-v1:" + digest.hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
        self.lock = threading.RLock()

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        values = [("Represent this sentence for searching relevant passages: " if query else "") + t for t in texts]
        output = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(values), 48):
                encoded = self.tokenizer(values[start:start + 48], padding=True, truncation=True,
                                         max_length=512, return_tensors="pt").to(self.device)
                vectors = self.model(**encoded).last_hidden_state[:, 0]
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
                output.append(vectors.cpu().numpy().astype(np.float32))
        return np.concatenate(output)
