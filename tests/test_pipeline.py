import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.pipeline import analyze, demo, write_outputs, load, COLUMNS, ROLES

class PipelineTests(unittest.TestCase):
    def test_demo_contract_and_reproducibility(self):
        frames=demo(); result=analyze(*frames,synthetic=True)
        self.assertEqual(result,analyze(*frames,synthetic=True))
        self.assertEqual(set(frames[0].gid),{n['gid'] for n in result['nodes']})
        self.assertGreaterEqual(len(result['top_nodes']),20)
        for n in result['nodes']:
            self.assertIn(n['role'],ROLES)
            for key in ['role_score','priority_score']: self.assertTrue(0<=n[key]<=1)
            self.assertTrue(0<len(n['evidence'])<=200)
            if n['depth']>=4 or n['is_seed']: self.assertNotIn(n['role'],['terminal','transit'])
        isolated=next(n for n in result['nodes'] if n['gid']=='KZ-1064')
        self.assertEqual(isolated['role'],'peripheral')
        self.assertIsNotNone(isolated['cluster_id'])

    def test_parquet_to_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            for name,frame in zip(['nodes','edges','transactions'],demo()): frame.to_parquet(Path(folder)/f'{name}.parquet',index=False)
            result=analyze(*load(folder)); write_outputs(result,folder)
            for name,columns in COLUMNS.items():
                frame=pd.read_csv(Path(folder)/f'{name}.csv')
                self.assertEqual(frame.columns.tolist(),columns)
                self.assertFalse(frame.isna().any().any())

    def test_terminal_boundary_and_seed(self):
        nodes=pd.DataFrame([['a',0,False],['b',4,False],['c',2,False],['d',1,True]],columns=['gid','depth','is_seed'])
        edges=pd.DataFrame([['a',n,100,1,1] for n in ['b','c','d']],columns=['src','dst','sum_kzt','n_tx','depth'])
        tx=pd.DataFrame([['a',n,'2026-09-01',100] for n in ['b','c','d']],columns=['src','dst','date','sum_kzt'])
        result={n['gid']:n for n in analyze(nodes,edges,tx)['nodes']}
        self.assertEqual(result['c']['role'],'terminal')
        self.assertEqual(result['b']['role'],'peripheral')
        self.assertEqual(result['d']['role'],'peripheral')

    def test_empty_edges(self):
        nodes,edges,tx=demo(); result=analyze(nodes,edges.iloc[:0],tx.iloc[:0])
        self.assertEqual(len(result['nodes']),len(nodes))
        self.assertEqual(len(result['clusters']),len(nodes))

    def test_unknown_node_rejected(self):
        n,e,t=demo(); e.loc[0,'src']='missing'
        with self.assertRaises(ValueError): analyze(n,e,t)

    def test_internal_volume(self):
        r=analyze(*demo()); membership={n['gid']:n['cluster_id'] for n in r['nodes']}
        for c in r['clusters']:
            expected=sum(e['sum_kzt'] for e in r['edges'] if membership[e['src']]==membership[e['dst']]==c['cluster_id'])
            self.assertEqual(c['sum_kzt_internal'],expected)

if __name__=='__main__': unittest.main()
