"""Literal source blocks and layout dependencies, never semantic facts."""
from dataclasses import dataclass, replace
import re

HEADING = re.compile(r'^ {0,3}(#{1,6})(?:[ \t]+|$)')
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
LIST = re.compile(r'^[ \t]*(?:[-+*]|[0-9]{1,9}[.)])[ \t]+\S')


def union_ranges(ranges):
    result = []
    for start, end in sorted(set(ranges)):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


@dataclass(frozen=True)
class Block:
    start: int
    end: int
    kind: str
    parents: tuple[int, ...]


@dataclass(frozen=True)
class BlockGraph:
    length: int
    blocks: tuple[Block, ...]

    def required(self, start, end, *, closure=True):
        if not 0 <= start < end <= self.length:
            raise ValueError('Seed outside original source')
        payload = tuple(i for i, block in enumerate(self.blocks)
                        if block.start < end and start < block.end)
        chosen, edges, pending = set(payload), set(), list(payload)
        if closure:
            while pending:
                child = pending.pop()
                for parent in self.blocks[child].parents:
                    if not 0 <= parent < child:
                        raise ValueError('Non-ancestral source block edge')
                    edges.add((child, parent))
                    if parent not in chosen:
                        chosen.add(parent)
                        pending.append(parent)
        ranges = union_ranges((self.blocks[i].start, self.blocks[i].end) for i in chosen)
        if not any(a <= start and b >= end for a, b in ranges):
            raise ValueError('Source blocks lost seed text')
        return ranges, payload, tuple(sorted(edges))


def source_blocks(text):
    # LF is the only layout separator; CRLF offsets remain literal.
    raw = text.split('\n')
    lines, offset = [], 0
    for n, value in enumerate(raw):
        end = offset+len(value)+(n+1 < len(raw))
        if end > offset:
            lines.append((offset, end, value[:-1] if value.endswith('\r') else value))
        offset = end
    blocks, stack, n = [], [], 0
    while n < len(lines):
        start, end, line = lines[n]
        if not line.strip():
            if blocks:
                blocks[-1] = replace(blocks[-1], end=end)
            n += 1
            continue
        if not blocks:
            start = 0
        heading, fence = HEADING.match(line), FENCE.match(line)
        parents = (stack[-1][1],) if stack else ()
        if heading:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            parents = (stack[-1][1],) if stack else ()
            kind = 'heading'
            stack.append((level, len(blocks)))
            n += 1
        elif fence and (fence.group(1)[0] != '`' or '`' not in fence.group(2)):
            marker = fence.group(1)
            kind, n = 'fence', n+1
            while n < len(lines):
                _, end, current = lines[n]
                n += 1
                close = FENCE.match(current)
                if close and close.group(1)[0] == marker[0] and len(close.group(1)) >= len(marker) and not close.group(2).strip():
                    break
        else:
            kind = 'list' if LIST.match(line) else 'paragraph'
            n += 1
            while n < len(lines):
                _, next_end, current = lines[n]
                if HEADING.match(current) or FENCE.match(current):
                    break
                if not current.strip():
                    if kind != 'list':
                        break
                    look = n+1
                    while look < len(lines) and not lines[look][2].strip():
                        look += 1
                    if look == len(lines) or not (LIST.match(lines[look][2]) or lines[look][2].startswith((' ', '\t'))):
                        break
                elif kind == 'paragraph' and LIST.match(current):
                    break
                end = next_end
                n += 1
            if kind == 'list' and blocks and blocks[-1].kind == 'paragraph':
                intro = blocks[-1]
                if text[intro.start:intro.end].rstrip().endswith((':', '：')):
                    parents = tuple(dict.fromkeys((*parents, len(blocks)-1)))
        blocks.append(Block(start, end, kind, parents))
    if not blocks and text:
        blocks.append(Block(0, len(text), 'whitespace', ()))
    if blocks:
        assert blocks[0].start == 0 and blocks[-1].end == len(text)
        assert all(left.end == right.start for left, right in zip(blocks, blocks[1:]))
    return BlockGraph(len(text), tuple(blocks))
