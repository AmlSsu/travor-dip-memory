import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from adaptive_evidence.core import Config, MemoryStore
from adaptive_evidence.source_prior import QUERY_DOCUMENT_MODES, query_document_prior, document_rank_mode
from test_contract import FakeEncoder

class QueryDocumentTests(unittest.TestCase):
    def test_back_reference_preserves_assistant_context_without_affecting_chat(self):
        self.assertEqual(document_rank_mode('query_document_adaptive','Put those details into a list',True),'hybrid')
        self.assertEqual(document_rank_mode('query_document_adaptive','整理一下上述内容',True),'hybrid')
        self.assertEqual(document_rank_mode('query_document_adaptive','Name two devices in the source',True),'user_hybrid')
        self.assertEqual(document_rank_mode('query_document_adaptive','Put those details into a list',False),'stem')

    def test_relevant_document_can_be_a_minority_of_scope(self):
        rows=[{'source_id':'document','role':'user','end':i*1000} for i in range(1,6)]
        rows += [{'source_id':str(i),'role':'user','end':600} for i in range(30)]
        index=SimpleNamespace(rows=rows)
        self.assertTrue(query_document_prior(index,np.arange(len(rows),0,-1)))
        scores=np.arange(len(rows),dtype=float);scores[0]=100
        self.assertFalse(query_document_prior(index,scores))

    def test_original_role_and_length_are_required(self):
        for role,length in [('assistant',10000),('user',1000)]:
            rows=[{'source_id':'s','role':role,'end':length} for _ in range(4)]
            self.assertFalse(query_document_prior(SimpleNamespace(rows=rows),np.arange(4)))

    def test_short_chat_exactly_preserves_baseline_with_one_query_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder())
            store.add(user_id='u',request_id='r',session_id='s',messages=[
                {'role':'user','content':'My violet unit is scheduled for Thursdays.'},
                {'role':'assistant','content':'I will check the violet schedule.'}])
            baseline=store.search(user_id='u',query='violet schedule',options=['Thursday','Friday'],mode='focus_stem_rules')
            for mode in QUERY_DOCUMENT_MODES:
                with patch.object(store.encoder,'encode',wraps=store.encoder.encode) as encode:
                    result=store.search(user_id='u',query='violet schedule',options=['Thursday','Friday'],mode=mode)
                self.assertEqual(result,baseline)
                self.assertEqual(encode.call_count,1)
                self.assertEqual(len(encode.call_args.args[0]),3)
                self.assertEqual(store.search(user_id='other',query='violet schedule',mode=mode),[])

    def test_document_links_are_source_only_and_budgeted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder(),Config(context_chars=4500))
            document='Section 1.2 Cobalt controls\nUse the seal specified in Section 9.3.\n'+('Cobalt maintenance background. '*200)+'\nSection 9.3 Seal\nCobalt requires a dry seal.'
            assistant='An earlier answer suggests a wet cobalt seal.'
            store.add(user_id='u',request_id='r',session_id='s',messages=[{'role':'user','content':document},{'role':'assistant','content':assistant}])
            for mode in QUERY_DOCUMENT_MODES:
                with patch.object(store.encoder,'encode',wraps=store.encoder.encode) as encode:
                    hits=store.search(user_id='u',query='cobalt maintenance seal controls',mode=mode)
                self.assertEqual(encode.call_count,1)
                self.assertLessEqual(sum(len(h['content']) for h in hits),4500)
                self.assertTrue(all(h['content'].split('] ',1)[1] in document or h['content'].split('] ',1)[1] in assistant for h in hits))

    def test_direct_document_lookup_filters_old_assistant_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder())
            original='Cobalt gate key is QuartzSail. The dry cobalt seal is required.\n'*100
            assistant='My previous answer recommended a wet cobalt seal.'
            store.add(user_id='u',request_id='r',session_id='s',messages=[{'role':'user','content':original},{'role':'assistant','content':assistant}])
            direct=store.search(user_id='u',query='Which cobalt seal is required?',mode='query_document_clean')
            self.assertTrue(direct)
            self.assertTrue(all(h['content'].startswith('[user]') for h in direct))
            followup=store.search(user_id='u',query='Repeat your answer about the cobalt seal',mode='query_document_clean')
            self.assertTrue(any(assistant in h['content'] for h in followup))
            self.assertTrue(any(row['role']=='assistant' for row in store.index('u').rows))
            self.assertGreater(store.document_direct_rankings,0)
            self.assertGreater(store.document_reference_rankings,0)

if __name__=='__main__':unittest.main()
