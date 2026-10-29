"""Relevant source evidence first, with bounded source diversity and context.

No inference, query generation, source rewriting, deletion or model calls.
All returned bodies are exact contiguous spans in the caller's own messages.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .core import content_tokens
from .evidence_windows import merged_rows


STANDING_RULE = re.compile(
    r"\b(?:always\s+(?:use|include|show|provide|write|format|keep|avoid)|"
    r"never\s+(?:include|show|use|write|reveal)|from\s+now\s+on|"
    r"for\s+(?:all|future)\s+(?:answers|responses|replies)|"
    r"(?:please\s+)?remember\s+to|make\s+sure\s+(?:to|you)|"
    r"please\s+(?:format|include|avoid))\b|"
    r"(?:以后|今后|始终|务必|每次回答|所有回答|请记住|回答时请)", re.I)


def pack_evidence(index, ranked, budget, top_k, render, *, query, reserve_rules=False,
                  coherent_sources=False, coherent_sections=False, source_links=False,
                  exclude_prior_assistant=False, direct_first=False):
    # RRF has bounded admission windows. Its long zero-score tail must not
    # turn evidence packing into an O(all_memories * context_size) operation.
    pool = [i for i in ranked[:max(200, min(512, top_k * 4))]
            if not exclude_prior_assistant or index.rows[i]['role'].casefold() != 'assistant']
    selected = []
    seen = set()
    chars = 0
    count = 0

    def include(i, cap):
        nonlocal chars, count
        if i in seen:
            return False
        if exclude_prior_assistant and index.rows[i]['role'].casefold() == 'assistant':
            return False
        merged = merged_rows(index, selected + [i])
        new_chars = sum(len(render(row)) for _, row in merged)
        if new_chars > cap or len(merged) > top_k:
            return False
        selected.append(i)
        seen.add(i)
        chars, count = new_chars, len(merged)
        return True

    # Preserve the best direct hit even when top_k is small or a rule exists.
    for i in pool:
        if include(i, budget):
            break
    if not selected:
        return []

    if reserve_rules and top_k >= 8:
        query_terms = content_tokens(query)
        ranked_positions = {i: p for p, i in enumerate(ranked)}
        rules = [i for i, row in enumerate(index.rows)
                 if row['role'].casefold() == 'user' and STANDING_RULE.search(row['content'])
                 and (i in pool or query_terms & index.terms[i])]
        rules.sort(key=lambda i: ranked_positions.get(i, len(ranked)))
        rule_cap = min(budget, chars + min(3000, budget // 8))
        for i in rules[:8]:
            include(i, rule_cap)

    def expand_source_context(source_cap, link_budget=None):
        nonlocal selected, chars, count
        # Source fidelity includes the dependencies around a matching passage.
        # Restore bounded complete originals as atomic groups before packing
        # unrelated fragments. This uses only stored offsets, never new text.
        source_chunks = {}
        for i, row in enumerate(index.rows):
            source_chunks.setdefault(row['source_id'], []).append(i)
        expanded = set()
        for seed in pool[:12] if coherent_sources or coherent_sections else []:
            source = index.rows[seed]['source_id']
            if source in expanded:
                continue
            expanded.add(source)
            indices = sorted(source_chunks[source], key=lambda i: index.rows[i]['start'])
            length = max(index.rows[i]['end'] for i in indices)
            if length <= 1000:
                continue
            if length > min(12000, budget // 2):
                if not coherent_sections:
                    continue
                position = indices.index(seed)
                indices = indices[max(0, position-2):position+3]
            proposal = selected + [i for i in indices if i not in seen]
            merged = merged_rows(index, proposal)
            new_chars = sum(len(render(row)) for _, row in merged)
            if new_chars > source_cap or len(merged) > top_k:
                continue
            selected = proposal
            seen.update(indices)
            chars, count = new_chars, len(merged)
            if len(expanded) >= 4:
                break

        if source_links:
            from .source_links import literal_links
            link_cap = link_budget if link_budget is not None else max(chars, int(budget * .70))
            for linked in literal_links(index, selected[:12]):
                include(linked, link_cap)
                for nearby in index.neighbors.get(linked, []):
                    if index.rows[nearby]['source_id'] == index.rows[linked]['source_id']:
                        include(nearby, link_cap)

    if not direct_first:
        expand_source_context(int(budget * .60))

    # An episode with many relevant chunks must not occupy the whole initial
    # evidence set. First admit distinct originals, then allow their tails.
    direct_cap = max(chars, int(budget * .80))
    direct_slots = max(count, max(1, int(top_k * .80)))
    sources = {index.rows[i]['source_id'] for i in selected}
    primary = []
    for i in pool:
        source = index.rows[i]['source_id']
        if source in sources:
            continue
        if include(i, direct_cap):
            sources.add(source)
            primary.append(i)
        if count >= direct_slots:
            break
    for i in pool:
        if count >= direct_slots:
            break
        if include(i, direct_cap):
            primary.append(i)

    if direct_first:
        # Direct evidence is protected before large local document windows.
        # Expansion remains bounded and always uses exact stored source spans.
        expand_source_context(budget, budget)

    # Same-message context gives long instructions/code a complete span;
    # adjacent messages supply local dialogue context. The direct hits keep
    # their relevance order, so earlier unrelated messages cannot jump ahead.
    seeds = list(dict.fromkeys(selected[:1] + primary))[:16]
    for seed in seeds:
        for i in index.neighbors.get(seed, []):
            include(i, budget)
    for i in pool:
        include(i, budget)

    output = []
    for _, row in merged_rows(index, selected):
        item = {'id':row['id'], 'content':render(row), 'score':1.0/(len(output)+1)}
        if row['timestamp'] is not None:
            item['created_at'] = datetime.fromtimestamp(row['timestamp']/1000, timezone.utc).isoformat()
        output.append(item)
    return output
