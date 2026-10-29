"""Optional source-linked retrieval views; independent experimental implementation.

Inspired by multi-key retrieval and public AML designs, not a reproduction of
their LLM fact extraction or graph algorithms. All views point to raw chunks.
"""
from __future__ import annotations

import sqlite3
import threading

import numpy as np

from .core import content_tokens, tokens, STOP


MODES = {"stem", "context", "diverse", "context_diverse", "stem_phrase", "user_stem", "user_hybrid"}


class RetrievalViews:
    def __init__(self, index):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE turns USING fts5(text, tokenize='porter unicode61')")
        self.db.execute("CREATE VIRTUAL TABLE windows USING fts5(text, tokenize='porter unicode61')")
        self.db.execute("CREATE VIRTUAL TABLE phrases USING fts5(text, tokenize='porter unicode61')")
        self.db.execute("CREATE VIRTUAL TABLE user_turns USING fts5(text, tokenize='porter unicode61')")
        pooled = []
        for i, row in enumerate(index.rows):
            # The base index only links neighbors within one ordered Add request.
            neighbors = index.neighbors[i]
            original = " ".join(sorted(content_tokens(row["content"])))
            context = " ".join(index.rows[j]["content"] for j in neighbors)
            self.db.execute("INSERT INTO turns(rowid,text) VALUES (?,?)", (i + 1, original))
            if row['role'].casefold() == 'user':
                self.db.execute("INSERT INTO user_turns(rowid,text) VALUES (?,?)", (i + 1, original))
            self.db.execute("INSERT INTO phrases(rowid,text) VALUES (?,?)",
                            (i + 1, ' '.join(t for t in tokens(row['content']) if t not in STOP)))
            self.db.execute("INSERT INTO windows(rowid,text) VALUES (?,?)",
                            (i + 1, " ".join(sorted(content_tokens(context)))))
            # Pool existing vectors, with double center weight. This is an
            # approximation to context encoding, with no extra model calls.
            vector = index.matrix[i] + index.matrix[neighbors].sum(axis=0)
            pooled.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        self.pooled = np.asarray(pooled, dtype=np.float32)
        self.db.commit()

    def lexical(self, text, table, limit):
        terms = sorted(content_tokens(text))[:96]
        if not terms:
            return []
        # Only tokenizer-produced alphanumeric/CJK terms enter MATCH syntax.
        expression = " OR ".join('"' + t + '"' for t in terms)
        if table not in {"turns", "windows", "user_turns"}:
            raise ValueError("Invalid view")
        with self.lock:
            return [int(row[0]) - 1 for row in self.db.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}),rowid LIMIT ?",
                (expression, limit))]

    def __del__(self):
        db = getattr(self, "db", None)
        if db is not None:
            db.close()

    def phrase_lexical(self, text, limit):
        ordered = [term for term in tokens(text) if term not in STOP][:96]
        phrases = list(dict.fromkeys(' '.join(ordered[i:i+2]) for i in range(len(ordered)-1)))
        if not phrases:
            return []
        expression = ' OR '.join('"' + phrase + '"' for phrase in phrases)
        with self.lock:
            return [int(row[0])-1 for row in self.db.execute(
                'SELECT rowid FROM phrases WHERE phrases MATCH ? ORDER BY bm25(phrases),rowid LIMIT ?',
                (expression,limit))]


def diverse_order(index, scores, order, limit=200):
    """Mild incremental redundancy penalty; never delete a stored memory."""
    pool = order[:limit]
    if not pool:
        return order
    relevance = scores[pool] / max(float(scores[pool].max()), 1e-12)
    similarity = np.clip(index.matrix[pool] @ index.matrix[pool].T, 0, 1)
    redundancy = np.zeros(len(pool))
    available = np.ones(len(pool), dtype=bool)
    selected = []
    for _ in pool:
        utility = .90 * relevance - .10 * redundancy
        utility[~available] = -np.inf
        winner = int(np.argmax(utility))
        selected.append(pool[winner])
        available[winner] = False
        redundancy = np.maximum(redundancy, similarity[winner])
    return selected + order[limit:]


def rank_views(index, queries, vectors, dense_scores, base_scores, mode, limit):
    scores = base_scores.copy()
    if mode != "diverse":
        with index.view_lock:
            if index.views is None:
                index.views = RetrievalViews(index)
            views = index.views
        for view, query in enumerate(queries):
            weight = 1.0 if view == 0 else .2 / (len(queries) - 1)
            streams = [] if mode == 'user_hybrid' else [(views.lexical(query, "turns", limit), .65)]
            if mode in {'user_stem','user_hybrid'}:
                # Original user statements add a provenance channel. Assistant
                # replies remain in the other streams and the final context.
                streams.append((views.lexical(query, 'user_turns', limit), .75))
            if mode == "stem_phrase":
                streams.append((views.phrase_lexical(query,limit), .55))
            if mode in {"context", "context_diverse"}:
                streams.extend([
                    (views.lexical(query, "windows", limit), .35),
                    (np.argsort(-(views.pooled @ vectors[view]), kind="stable")[:limit], .35),
                ])
            for order, channel_weight in streams:
                for rank, i in enumerate(order, 1):
                    scores[i] += weight * channel_weight / (60 + rank)
    order = list(map(int, np.lexsort((-dense_scores, -scores))))
    if mode in {"diverse", "context_diverse"}:
        order = diverse_order(index, scores, order, limit)
    return order
