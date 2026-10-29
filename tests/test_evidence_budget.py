import unittest
from types import SimpleNamespace

from adaptive_evidence.focused_packing import pack_evidence


class EvidenceBudgetTests(unittest.TestCase):
    def test_large_source_expansion_cannot_displace_second_direct_evidence(self):
        source_a=''.join(chr(65+i)*300 for i in range(5))
        source_b='Second direct fact: the amber valve needs a dry seal. '+('z'*1745)
        rows=[]
        for i in range(5):
            rows.append({'id':str(i),'source_id':'a','start':i*300,'end':(i+1)*300,
                         'content':source_a[i*300:(i+1)*300],'role':'user','timestamp':None})
        rows.append({'id':'5','source_id':'b','start':0,'end':len(source_b),
                     'content':source_b,'role':'assistant','timestamp':1700000000000})
        index=SimpleNamespace(rows=rows,neighbors={})
        render=lambda row:'['+row['role']+'] '+row['content']
        args=(index,[0,5,1,2,3,4],3000,20,render)
        old=pack_evidence(*args,query='amber valve',coherent_sources=True)
        new=pack_evidence(*args,query='amber valve',coherent_sources=True,direct_first=True)
        self.assertFalse(any('Second direct fact' in h['content'] for h in old))
        self.assertTrue(any('Second direct fact' in h['content'] for h in new))
        self.assertTrue(any(h.get('created_at') for h in new))
        self.assertEqual(new[0]['content'].split('] ',1)[1][0],'A')
        self.assertLessEqual(sum(len(h['content']) for h in new),3000)
        self.assertTrue(all(h['content'].split('] ',1)[1] in source_a or
                            h['content'].split('] ',1)[1] in source_b for h in new))

    def test_direct_first_preserves_top_k_and_an_oversized_source_is_not_truncated(self):
        row={'id':'large','source_id':'a','start':0,'end':800,
             'content':'v'*800,'role':'user','timestamp':None}
        index=SimpleNamespace(rows=[row],neighbors={})
        render=lambda r:'[user] '+r['content']
        self.assertEqual(pack_evidence(index,[0],700,1,render,query='v',direct_first=True),[])
        hits=pack_evidence(index,[0],900,1,render,query='v',direct_first=True)
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['content'],'[user] '+row['content'])


if __name__=='__main__':unittest.main()
