"""Literal within-source section references, without semantic inference."""
import re

ID = r'(?:[IVXLCDM]+(?:\.[A-Z0-9]+)+|\d+(?:\.[A-Z0-9]+)*)'
REFERENCE = re.compile(r'\b(?:section|sec\.?|clause|article|rule)\s+('+ID+r')\b',re.I)
HEADING = re.compile(r'(?m)^\s*(?:#{1,6}\s*|\*\*)?(?:(?:section|sec\.?|clause|article|rule)\s+)?('+ID+r')(?=[\s:.)])',re.I)

def literal_links(index, seeds):
    """Follow only an explicit reference to a heading in the same original."""
    source_headings={}
    for i,row in enumerate(index.rows):
        headings=source_headings.setdefault(row['source_id'],{})
        for match in HEADING.finditer(row['content']):
            headings.setdefault(match.group(1).casefold(),[]).append(i)
    result=[]
    for seed in seeds:
        row=index.rows[seed];headings=source_headings[row['source_id']]
        for reference in REFERENCE.finditer(row['content']):
            result.extend(headings.get(reference.group(1).casefold(),[])[:2])
    return list(dict.fromkeys(result))[:16]
