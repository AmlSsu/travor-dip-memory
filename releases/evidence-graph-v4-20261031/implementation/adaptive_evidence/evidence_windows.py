"""Budgeted original-source windows; no model, inferred facts or answer text.

Reserve space for neighboring evidence before the context fills. Merge only
overlapping/adjacent offsets in the same message and preserve gaps as separate
items. Relevance chooses windows; known Add message order orders each window.
"""
from __future__ import annotations

from collections import defaultdict

from .core import digest


def merged_rows(index, selected):
    by_source = defaultdict(list)
    priority = {i: rank for rank, i in enumerate(selected)}
    for i in selected:
        by_source[index.rows[i]["source_id"]].append(i)
    output = []
    for indices in by_source.values():
        indices.sort(key=lambda i: index.rows[i]["start"])
        row = None
        members = []
        for i in indices:
            current = index.rows[i]
            if row is None or current["start"] > row["end"]:
                if row is not None:
                    output.append((min(priority[j] for j in members), row))
                row = dict(current)
                members = [i]
            else:
                if current["end"] > row["end"]:
                    row["content"] += current["content"][row["end"] - current["start"]:]
                    row["end"] = current["end"]
                    row["id"] = digest([row["source_id"], row["start"], row["end"]])
                members.append(i)
        if row is not None:
            output.append((min(priority[j] for j in members), row))
    return sorted(output, key=lambda pair: pair[0])


def select_evidence(index, ranked, budget, top_k, render, *, chronological=True):
    selected = []
    seen = set()

    def cost(values):
        rows = merged_rows(index, values)
        return sum(len(render(row)) for _, row in rows), len(rows)

    def include(i, cap):
        if i in seen:
            return False
        chars, count = cost(selected + [i])
        if chars > cap or count > top_k:
            return False
        selected.append(i)
        seen.add(i)
        return True

    # Compute the reserve using actual rendered characters, not top_k. Most
    # direct hits stay; the reserve can be reclaimed when no neighbors fit.
    direct_cap = max(1, int(budget * .75))
    for i in ranked:
        include(i, direct_cap)
        if cost(selected)[1] >= max(1, int(top_k * .75)):
            break
    if not selected:
        for i in ranked:
            if include(i, budget):
                break
    seeds = list(selected)
    for seed in seeds[:20]:
        for i in index.neighbors.get(seed, []):
            include(i, budget)
    for i in ranked:
        include(i, budget)

    rows = merged_rows(index, selected)
    # Only explicitly ordered messages within the same Add can be reordered.
    # Keep groups in first-hit order; never sort opaque session/request IDs.
    group_priority = {}
    for rank, row in rows:
        key = (row["session_id"], row["request_id"])
        group_priority.setdefault(key, rank)
    if chronological:
        rows.sort(key=lambda pair: (group_priority[(pair[1]["session_id"], pair[1]["request_id"])],
                                   pair[1]["position"], pair[1]["start"]))
    output = []
    for _, row in rows:
        item = {"id": row["id"], "content": render(row), "score": 1.0 / (len(output) + 1)}
        if row["timestamp"] is not None:
            from datetime import datetime, timezone
            item["created_at"] = datetime.fromtimestamp(row["timestamp"] / 1000, timezone.utc).isoformat()
        output.append(item)
    return output
