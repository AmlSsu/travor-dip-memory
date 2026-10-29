import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_evidence.core import Config, MemoryStore
from test_contract import FakeEncoder


class EvidenceWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name)/'test.db', FakeEncoder(),
                                 Config(context_chars=900, chunk_chars=300, overlap_chars=60))

    def tearDown(self):
        self.tmp.cleanup()

    def test_neighbor_reserve_survives_large_top_k(self):
        messages = [{'role':'user','content':'The launch venue is the observatory. '+('topic '*23)},
                    {'role':'user','content':'It starts at 19:45 next Thursday.'}]
        messages += [{'role':'user','content':f'Other launch discussion {i}. '+('topic '*23)} for i in range(12)]
        self.store.add(user_id='a',request_id='r',session_id='s',messages=messages)
        index=self.store.index('a')
        ordered=sorted(range(len(index.rows)),key=lambda i:(index.rows[i]['position']==1,index.rows[i]['position']))
        with patch.object(self.store,'ranked',return_value=ordered):
            old=self.store.search(user_id='a',query='launch venue time',mode='hybrid_window')
            new=self.store.search(user_id='a',query='launch venue time',mode='evidence_window')
        self.assertNotIn('19:45',str(old))
        self.assertIn('19:45',str(new))
        self.assertLessEqual(sum(len(h['content']) for h in new),900)

    def test_overlapping_spans_merge_exactly_without_repeating_source(self):
        source=''.join(f'Original line {i:03}: blue green orange.\n' for i in range(15))
        self.store.add(user_id='a',request_id='r',session_id='s',messages=[{'role':'user','content':source}])
        hits=self.store.search(user_id='a',query='blue',mode='evidence_window')
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['content'],'[user] '+source)

    def test_independent_messages_with_same_text_keep_dates(self):
        for name,stamp in [('a',1704067200000),('b',1735689600000)]:
            self.store.add(user_id='a',request_id=name,session_id=name,messages=[{'role':'user','content':'My base is Oslo.','timestamp':stamp}])
        hits=self.store.search(user_id='a',query='base',mode='evidence_window')
        self.assertEqual(len(hits),2)
        self.assertEqual({h['created_at'][:4] for h in hits},{'2024','2025'})
        self.assertEqual(self.store.search(user_id='other',query='base',mode='evidence_window'),[])

    def test_top_one_keeps_best_direct_hit_and_original_text(self):
        self.store.add(user_id='a',request_id='r',session_id='s',messages=[{'role':'user','content':'unrelated greeting'},{'role':'user','content':'violet train platform 18'}])
        hit=self.store.search(user_id='a',query='violet train',top_k=1,mode='evidence_window')
        self.assertEqual(len(hit),1)
        self.assertEqual(hit[0]['content'],'[user] violet train platform 18')

    def test_chronology_is_preserved_within_one_add(self):
        self.store.add(user_id='a',request_id='r',session_id='s',messages=[{'role':'user','content':'Original plan: Monday.'},{'role':'user','content':'Correction: Tuesday replaces Monday.'}])
        hits=self.store.search(user_id='a',query='Correction Tuesday',mode='evidence_window')
        self.assertIn('Original plan',hits[0]['content'])
        self.assertIn('Correction',hits[1]['content'])


if __name__=='__main__':unittest.main()
