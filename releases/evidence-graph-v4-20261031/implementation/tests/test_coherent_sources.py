import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_evidence.core import Config, MemoryStore
from test_contract import FakeEncoder


class CoherentSourceTests(unittest.TestCase):
    def test_middle_match_restores_original_dependencies_within_budget(self):
        original = ('Prerequisite: disconnect the amber supply before maintenance.\n' +
                    'Background specifications and notes.\n' * 30 +
                    'Step 4: turn the cobalt latch clockwise.\n' +
                    'Verification and audit details.\n' * 25 +
                    'Final requirement: reconnect amber only after the dry seal passes inspection.')
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp)/'test.db', FakeEncoder(), Config(context_chars=6500))
            store.add(user_id='u',request_id='r',session_id='s',messages=[
                {'role':'user','content':original},
                {'role':'user','content':'Unrelated cobalt latch history. '*70}])
            index=store.index('u')
            seed=next(i for i,r in enumerate(index.rows) if 'Step 4:' in r['content'])
            order=[seed]+[i for i in range(len(index.rows)) if i!=seed]
            for mode in ['coherent_stem','coherent_hybrid','coherent_sections','coherent_links']:
                with patch.object(store,'ranked',return_value=order):
                    hits=store.search(user_id='u',query='cobalt latch maintenance',mode=mode)
                self.assertEqual(hits[0]['content'],'[user] '+original)
                self.assertLessEqual(sum(len(h['content']) for h in hits),6500)
                self.assertEqual(store.search(user_id='other',query='cobalt latch',mode=mode),[])

    def test_overbudget_source_stays_literal_and_does_not_hide_other_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder(),Config(context_chars=1500,chunk_chars=400,overlap_chars=80))
            original='Violet chamber details. '*120+'The violet seal requires calibration.'
            store.add(user_id='u',request_id='r',session_id='s',messages=[
                {'role':'user','content':original},
                {'role':'user','content':'Violet seal calibration interval is 12 days.'}])
            for mode in ['coherent_stem','coherent_hybrid','coherent_sections','coherent_links']:
                hits=store.search(user_id='u',query='violet seal calibration',mode=mode)
                self.assertTrue(any('12 days' in h['content'] for h in hits))
                self.assertTrue(all(h['content'][7:] in original or h['content'][7:]=='Violet seal calibration interval is 12 days.' for h in hits))
                self.assertLessEqual(sum(len(h['content']) for h in hits),1500)

    def test_coherence_reuses_the_single_query_embedding_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder())
            store.add(user_id='u',request_id='r',session_id='s',messages=[{'role':'user','content':'Amber maintenance requires the cobalt latch. '*60}])
            for mode in ['coherent_stem','coherent_hybrid','coherent_sections','coherent_links','focus_user_rules','document_focus']:
                with patch.object(store.encoder,'encode',wraps=store.encoder.encode) as encode:
                    store.search(user_id='u',query='amber maintenance',options=['dry seal','wet seal'],mode=mode)
                self.assertEqual(encode.call_count,1)
                self.assertEqual(len(encode.call_args.args[0]),3)
                self.assertTrue(encode.call_args.kwargs['query'])

    def test_literal_reference_never_crosses_original_sources(self):
        from adaptive_evidence.source_links import literal_links
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder(),Config(chunk_chars=240,overlap_chars=40))
            own='Section 2.1 Controls\nThe amber switch requires the seal specified in Section 9.3.\n'+'Ordinary background. '*80+'\nSection 9.3 Seal\nUse the dry seal for the amber switch.'
            other='Section 9.3 Seal\nUse the wet seal for the unrelated violet device.'
            store.add(user_id='u',request_id='r',session_id='s',messages=[{'role':'user','content':own},{'role':'user','content':other}])
            index=store.index('u');seed=next(i for i,r in enumerate(index.rows) if 'specified in Section 9.3' in r['content'])
            links=literal_links(index,[seed])
            self.assertTrue(any('Use the dry seal' in index.rows[i]['content'] for i in links))
            self.assertTrue(all(index.rows[i]['source_id']==index.rows[seed]['source_id'] for i in links))

    def test_user_provenance_prefers_original_but_keeps_assistant_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder())
            store.add(user_id='u',request_id='r',session_id='s',messages=[
                {'role':'assistant','content':'The cobalt gate key is RavenWheel.'},
                {'role':'user','content':'The cobalt gate key is QuartzSail.'}])
            hits=store.search(user_id='u',query='cobalt gate key',mode='focus_user_rules')
            self.assertTrue(hits[0]['content'].startswith('[user]'))
            self.assertTrue(any('RavenWheel' in h['content'] for h in hits))
            self.assertTrue(any('QuartzSail' in h['content'] for h in hits))
            self.assertEqual(store.search(user_id='other',query='cobalt gate key',mode='focus_user_rules'),[])

    def test_document_prior_preserves_ordinary_conversation_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder())
            store.add(user_id='u',request_id='r',session_id='s',messages=[
                {'role':'assistant','content':'The violet unit runs nightly.'},
                {'role':'user','content':'My violet unit is scheduled for Thursdays.'}])
            baseline=store.search(user_id='u',query='violet unit schedule',mode='focus_stem_rules')
            self.assertEqual(store.search(user_id='u',query='violet unit schedule',mode='document_focus'),baseline)

    def test_dominant_document_prior_does_not_discard_assistant_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'test.db',FakeEncoder())
            document='Cobalt seal instruction: use the dry seal.\n'+'Ordinary background specifications.\n'*400
            store.add(user_id='u',request_id='r',session_id='s',messages=[
                {'role':'assistant','content':'Cobalt seal instruction: use the wet seal.'},
                {'role':'user','content':document}])
            hits=store.search(user_id='u',query='cobalt seal instruction',mode='document_focus')
            self.assertTrue(hits[0]['content'].startswith('[user]'))
            self.assertTrue(any('wet seal' in h['content'] for h in hits))
            self.assertTrue(any('dry seal' in h['content'] for h in hits))


if __name__=='__main__':unittest.main()
