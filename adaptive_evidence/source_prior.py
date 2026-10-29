"""Current-query source coherence; sizes and existing scores only."""
from collections import Counter
import re
import numpy as np

QUERY_DOCUMENT_MODES={'query_document_focus','query_document_links','query_document_hybrid','query_document_user_hybrid','query_document_adaptive','query_document_clean','query_document_evidence'}
BACK_REFERENCE=re.compile(r'\b(?:those|these|them|previous(?:ly)?|earlier|above|again|continue|repeat)\b|\byour\s+(?:answer|response|reply|explanation|suggestions?|recommendations?|advice)\b|\byou\s+(?:said|told|mentioned|explained)\b|(?:刚才|上面|上述|这些|那些|前面|之前的|你的回答|继续)',re.I)

def document_rank_mode(mode,query,active):
    if not active:return 'stem'
    if mode=='query_document_hybrid':return 'hybrid'
    if mode in {'query_document_adaptive','query_document_clean','query_document_evidence'}:return 'hybrid' if BACK_REFERENCE.search(query) else 'user_hybrid'
    return 'user_hybrid' if mode=='query_document_user_hybrid' else 'user_stem'

def query_document_prior(index, dense_scores):
    lengths={}
    user_sources=set()
    for row in index.rows:
        source=row['source_id']
        lengths[source]=max(lengths.get(source,0),row['end'])
        if row['role'].casefold()=='user':user_sources.add(source)
    counts=Counter(index.rows[int(i)]['source_id'] for i in np.argsort(-dense_scores,kind='stable')[:12])
    return any(lengths[source]>=2000 and counts[source]>=3 for source in user_sources)
