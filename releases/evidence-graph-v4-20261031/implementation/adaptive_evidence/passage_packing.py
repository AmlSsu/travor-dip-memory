"""Fine source passages ranked without generating or embedding new text."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .core import content_tokens, digest, split_spans
from .evidence_windows import merged_rows
from .focused_packing import STANDING_RULE


class Passages:
    def __init__(self, index):
        self.rows, self.parents, self.terms = [], [], []
        self.neighbors = {}
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE passages USING fts5(text, tokenize='porter unicode61')")
        seen = set()
        for parent, row in enumerate(index.rows):
            siblings = []
            for start, end in split_spans(row['content'], 420, 60):
                key = (row['source_id'], row['start'] + start, row['start'] + end)
                if key in seen:
                    continue
                seen.add(key)
                item = {**row, 'start': key[1], 'end': key[2],
                        'content': row['content'][start:end], 'id': digest(key)}
                i = len(self.rows)
                self.rows.append(item); self.parents.append(parent)
                terms = content_tokens(item['content']); self.terms.append(terms)
                self.db.execute('INSERT INTO passages(rowid,text) VALUES (?,?)',
                                (i + 1, ' '.join(sorted(terms))))
                siblings.append(i)
            for position, i in enumerate(siblings):
                self.neighbors[i] = siblings[max(0, position - 1):position + 2]
        self.db.commit()

    def close(self):
        self.db.close()

    def __del__(self):
        db = getattr(self, 'db', None)
        if db is not None:
            db.close()


def pack_passages(index, ranked, budget, top_k, render, *, query, reserve_rules=False):
    # The existing query embedding is already reflected in parent rank. The
    # fine view is an in-memory lexical projection of those same raw sources.
    with index.view_lock:
        if getattr(index, 'passages', None) is None:
            index.passages = Passages(index)
        view = index.passages
        terms = sorted(content_tokens(query))[:96]
        expression = ' OR '.join('"' + term + '"' for term in terms)
        lexical = ([int(r[0]) - 1 for r in view.db.execute(
            'SELECT rowid FROM passages WHERE passages MATCH ? ORDER BY bm25(passages),rowid LIMIT 600',
            (expression,))] if expression else [])
    parent_rank = {i: rank for rank, i in enumerate(ranked, 1)}
    lexical_rank = {i: rank for rank, i in enumerate(lexical, 1)}
    score = {i: 1 / (60 + parent_rank[parent]) +
             (.9 / (60 + lexical_rank[i]) if i in lexical_rank else 0)
             for i, parent in enumerate(view.parents)}
    pool = sorted(score, key=lambda i: (-score[i], parent_rank[view.parents[i]], i))[:600]
    selected, seen, redundant = [], set(), set()

    def include(i, cap):
        if i in seen:
            return False
        merged = merged_rows(view, selected + [i])
        if len(merged) > top_k or sum(len(render(row)) for _, row in merged) > cap:
            return False
        selected.append(i); seen.add(i)
        return True

    if pool:
        include(pool[0], budget)
    if reserve_rules and top_k >= 8:
        cap = min(budget, sum(len(render(view.rows[i])) for i in selected) + min(1500, budget // 12))
        rules = [i for i, row in enumerate(view.rows)
                 if row['role'].casefold() == 'user' and STANDING_RULE.search(row['content'])]
        for i in sorted(rules, key=lambda i: -score[i])[:6]:
            include(i, cap)

    # Discount nearly repeated passages instead of allowing repeated long
    # explanations to crowd out independent evidence. This is deterministic.
    bounded_ids = set(pool) | set(selected)
    vocab = {term: j for j, term in enumerate(sorted(set().union(*(view.terms[i] for i in bounded_ids))))}
    masks = {i: sum(1 << vocab[term] for term in view.terms[i]) for i in bounded_ids}
    similarities = {i: 0.0 for i in pool}
    def discount(seed):
        for i in pool:
            similarity = (masks[i] & masks[seed]).bit_count() / max(1, (masks[i] | masks[seed]).bit_count())
            similarities[i] = max(similarities[i], similarity)
    for seed in selected:
        discount(seed)
    while pool:
        best, utility = None, float('-inf')
        for i in pool:
            if i in seen or i in redundant:
                continue
            candidate = score[i] * (1 - .45 * similarities[i])
            if candidate > utility:
                best, utility = i, candidate
        if best is None:
            break
        pool.remove(best)
        if include(best, int(budget * .90)):
            discount(best)
        else:
            redundant.add(best)
        if len(merged_rows(view, selected)) >= max(1, int(top_k * .90)):
            break
    # Bounded local source context preserves references around selected spans.
    for seed in selected[:24]:
        for i in view.neighbors.get(seed, []):
            include(i, budget)
    for i in sorted(score, key=lambda i: -score[i])[:600]:
        include(i, budget)
    output = []
    for _, row in merged_rows(view, selected):
        item = {'id': row['id'], 'content': render(row), 'score': 1 / (len(output) + 1)}
        if row['timestamp'] is not None:
            item['created_at'] = datetime.fromtimestamp(row['timestamp'] / 1000, timezone.utc).isoformat()
        output.append(item)
    return output
