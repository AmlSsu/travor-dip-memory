"""Select exact source ranges with shared layout closure and joint union cost."""
from collections import defaultdict
from datetime import datetime, timezone
import math

from .core import content_tokens, digest
from .focused_packing import STANDING_RULE
from .block_graph import source_blocks, union_ranges


def direct_rows(index, entries):
    """Entries are (source,start,end); only literal touching ranges coalesce."""
    sources, templates = defaultdict(list), {}
    for row in index.rows:
        templates.setdefault(row['source_id'], row)
    for priority, (source, start, end) in enumerate(entries):
        original = index.source_texts[source]
        if not 0 <= start < end <= len(original):
            raise ValueError('Direct span outside source')
        sources[source].append((start, end, priority))
    output = []
    for source, spans in sources.items():
        for start, end in union_ranges((a, b) for a, b, _ in spans):
            priority = min(p for a, b, p in spans if a < end and start < b)
            row = dict(templates[source], start=start, end=end,
                       id=digest([source, start, end]), content=index.source_texts[source][start:end])
            output.append((priority, row))
    return sorted(output, key=lambda item: (item[0], item[1]['start']))


def select_direct_blocks(index, ranked, budget, top_k, render, *, query, closure=True, joint=True):
    pool = list(dict.fromkeys(ranked[:80]))
    position = {i: n for n, i in enumerate(ranked)}
    terms = content_tokens(query)
    raw = {t: math.log(1+(index.bm25.count+1)/(1+len(index.bm25.postings.get(t, ())))) for t in sorted(terms)}
    denominator = sum(raw.values()) or 1
    weights = {t: v/denominator for t, v in raw.items()}
    trees, proposals, groups, group_proposals, source_cache = {}, [], [], defaultdict(list), {}
    templates = {}
    for i, row in enumerate(index.rows):
        source = row['source_id']
        templates.setdefault(source, row)
        if source not in index.source_texts or index.source_texts[source][row['start']:row['end']] != row['content']:
            raise ValueError('Indexed chunk differs from original source')

    def rows(values):
        by_source = defaultdict(list)
        for source, a, b in values:
            by_source[source].append((a, b))
        output = []
        for source, spans in by_source.items():
            key = source, union_ranges(spans)
            if key not in source_cache:
                source_cache[key] = [dict(templates[source], start=a, end=b,
                    id=digest([source, a, b]), content=index.source_texts[source][a:b]) for a, b in key[1]]
            output.extend(source_cache[key])
        return output

    def cost(values):
        rendered = rows(values)
        return sum(len(render(row)) for row in rendered), len(rendered)

    def features(values):
        return set().union(*(content_tokens(row['content']) & terms for row in rows(values)))

    def covered_ranges(group, values):
        present = defaultdict(list)
        for source, a, b in values:
            present[source].append((a, b))
        return all(any(c <= a and d >= b for c, d in union_ranges(present[source])) for source, a, b in group)

    for seed in pool:
        row = index.rows[seed]
        source = row['source_id']
        if source not in trees:
            trees[source] = source_blocks(index.source_texts[source])
        ranges, payload, edges = trees[source].required(row['start'], row['end'], closure=closure)
        group = tuple((source, a, b) for a, b in ranges)
        proposals.append(dict(seed=seed, source_id=source, required_ranges=ranges,
                              payload_blocks=payload, dependency_edges=edges))
        if group not in group_proposals:
            groups.append(group)
        group_proposals[group].append(len(proposals)-1)
    audit = dict(fallback=None, pool=len(pool), groups=len(groups), block_proposals=proposals,
        block_sources={s: [dict(start=b.start, end=b.end, kind=b.kind, parents=b.parents) for b in tree.blocks]
                       for s, tree in trees.items()}, closure=closure, joint=joint, semantic_edges=0,
        representation='direct_original_blocks', projected_extra_characters=0)
    if not groups:
        audit['fallback'] = 'empty_pool'
        return [], audit
    if cost(groups[0])[0] > budget or cost(groups[0])[1] > top_k:
        audit['fallback'] = 'anchor_block_closure_does_not_fit'
        return [], audit
    selected, admitted = list(groups[0]), [groups[0]]
    protected = []
    if top_k >= 8:
        rule_pool = set(ranked[:max(200, min(512, top_k*4))])
        rules = [i for i, row in enumerate(index.rows) if row['role'].casefold() == 'user'
                 and STANDING_RULE.search(row['content']) and (i in rule_pool or terms & index.terms[i])]
        rules.sort(key=lambda i: position.get(i, len(ranked)))
        cap = min(budget, cost(selected)[0]+min(3000, budget//8))
        for seed in rules[:8]:
            row = index.rows[seed]
            entry = row['source_id'], row['start'], row['end']
            chars, count = cost(selected+[entry])
            if chars <= cap and count <= top_k:
                if entry not in selected:
                    selected.append(entry)
                protected.append(seed)
    group_features = {g: features(g) for g in groups}
    ranks = {g: min(position[proposals[p]['seed']] for p in group_proposals[g]) for g in groups}
    covered, remaining = features(selected), [g for g in groups if g != groups[0]]
    while remaining:
        old_chars, old_count = cost(selected)
        choices = []
        for n, group in enumerate(remaining):
            chars, count = cost(selected+list(group))
            if covered_ranges(group, selected):
                continue
            if chars > budget or count > top_k:
                continue
            rank = ranks[group]
            gain = 3/(1+rank/8)+1.5*sum(weights[t] for t in sorted(group_features[group]-covered))
            priority = gain/math.sqrt(max(250, chars-old_chars)) if joint else -rank
            choices.append((priority, -rank, -n, group))
        if not choices:
            break
        _, _, negative_n, group = max(choices, key=lambda choice: choice[:3])
        selected.extend(entry for entry in group if entry not in selected)
        admitted.append(group)
        covered |= group_features[group]
        remaining.pop(-negative_n)
    audit.update(selected_range_groups=[[dict(source_id=s, start=a, end=b) for s, a, b in g] for g in admitted],
        admitted_proposal_indices=[group_proposals[g] for g in admitted],
        unadmitted_groups=[dict(ranges=g, reason='already_covered' if covered_ranges(g, selected) else 'budget_or_top_k') for g in remaining],
        protected_rule_chunks=protected, covered_query_terms=len(covered), query_terms=len(terms),
        rendered_characters=cost(selected)[0], rendered_items=cost(selected)[1])
    return selected, audit


def pack_direct_blocks(index, ranked, budget, top_k, render, *, query, variant='full'):
    if variant not in {'full', 'no_block_closure', 'no_joint'}:
        raise ValueError('Unknown direct block variant')
    selected, audit = select_direct_blocks(index, ranked, budget, top_k, render, query=query,
        closure=variant != 'no_block_closure', joint=variant != 'no_joint')
    output = []
    for _, row in direct_rows(index, selected):
        item = dict(id=row['id'], content=render(row), score=1/(len(output)+1))
        if row['timestamp'] is not None:
            item['created_at'] = datetime.fromtimestamp(row['timestamp']/1000, timezone.utc).isoformat()
        output.append(item)
    return output, audit
