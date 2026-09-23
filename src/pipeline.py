import argparse
import json
from pathlib import Path
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROLES = ['consolidator', 'distributor', 'transit', 'terminal', 'coordinator', 'peripheral']
COLUMNS = {
    'nodes_roles': ['gid', 'role', 'role_score', 'cluster_id', 'priority_score', 'evidence'],
    'clusters': ['cluster_id', 'n_nodes', 'n_seed', 'sum_kzt_internal', 'top_gids', 'hypothesis'],
    'top_nodes': ['rank', 'gid', 'role', 'priority_score', 'why'],
}

def demo():
    import random
    r = random.Random(42)
    nodes = pd.DataFrame([{'gid': f'KZ-{1000+i}', 'depth': min(4, i % 5), 'is_seed': i in (0, 16, 32, 48)} for i in range(65)])
    rows = []
    for group in range(4):
        base = group * 16
        for i in range(1, 16):
            src, dst = (base+i, base) if i < 8 else (base, base+i)
            rows.append([f'KZ-{1000+src}', f'KZ-{1000+dst}', r.randint(150, 950)*10000, r.randint(2, 14), min(4, i%5)])
        for i in range(2, 14, 3):
            rows.append([f'KZ-{1000+base+i}', f'KZ-{1001+base+i}', r.randint(100, 500)*10000, 3, 2])
    for a,b in [(0,16),(16,32),(32,48),(48,0),(6,24),(27,45)]:
        rows.append([f'KZ-{1000+a}', f'KZ-{1000+b}', r.randint(300,900)*10000, 5, 1])
    edges = pd.DataFrame(rows, columns=['src','dst','sum_kzt','n_tx','depth'])
    tx=[]
    for e in edges.to_dict('records'):
        for j in range(e['n_tx']):
            tx.append({'src':e['src'], 'dst':e['dst'], 'date':f'2026-09-{1+j:02d}', 'sum_kzt':e['sum_kzt']/e['n_tx']})
    return nodes, edges, pd.DataFrame(tx)

def validate(nodes, edges, tx):
    for df, fields in [(nodes,['gid','depth','is_seed']), (edges,['src','dst','sum_kzt','n_tx','depth']), (tx,['src','dst','date','sum_kzt'])]:
        missing = set(fields)-set(df.columns)
        if missing: raise ValueError('Отсутствуют поля: '+', '.join(sorted(missing)))
        if df[fields].isna().any().any(): raise ValueError('Обязательные поля содержат пустые значения')
    nodes['gid'] = nodes['gid'].astype(str)
    if nodes.gid.duplicated().any(): raise ValueError('gid должен быть уникальным')
    if not nodes.is_seed.isin([True,False,0,1]).all(): raise ValueError('is_seed должен быть boolean')
    nodes['is_seed'] = nodes.is_seed.astype(bool)
    for frame in [nodes,edges]:
        depth=pd.to_numeric(frame['depth'],errors='raise')
        if ((depth<0)|(depth%1!=0)).any(): raise ValueError('depth должен быть целым и неотрицательным')
        frame['depth']=depth.astype(int)
    for df in [edges,tx]:
        for c in ['src','dst']: df[c]=df[c].astype(str)
        if not (set(df.src)|set(df.dst)) <= set(nodes.gid): raise ValueError('Ребро ссылается на неизвестный gid')
        df['sum_kzt']=pd.to_numeric(df.sum_kzt,errors='raise')
        if ((df.sum_kzt<0)|(~df.sum_kzt.map(lambda x: __import__('math').isfinite(x)))).any(): raise ValueError('Некорректная сумма')
    edges['n_tx']=pd.to_numeric(edges.n_tx, errors='raise')
    if ((edges.n_tx<1)|(edges.n_tx%1!=0)).any(): raise ValueError('n_tx должен быть положительным целым')
    tx['date']=pd.to_datetime(tx.date, errors='raise')

def analyze(nodes, edges, tx, config=None, synthetic=False):
    nodes,edges,tx=nodes.copy(),edges.copy(),tx.copy()
    validate(nodes,edges,tx)
    cfg=config or json.loads((ROOT/'config.json').read_text())
    g=nx.DiGraph()
    g.add_nodes_from(sorted(nodes.gid))
    for e in edges.sort_values(['src','dst']).to_dict('records'):
        old=g.get_edge_data(e['src'],e['dst'],default={})
        g.add_edge(e['src'],e['dst'],weight=old.get('weight',0)+float(e['sum_kzt']),n_tx=old.get('n_tx',0)+int(e['n_tx']))
    u=nx.Graph(); u.add_nodes_from(g)
    for a,b,d in g.edges(data=True):
        if d['weight']>0: u.add_edge(a,b,weight=u.get_edge_data(a,b,default={}).get('weight',0)+d['weight'])
    communities=nx.community.louvain_communities(u, seed=cfg['seed'], weight='weight') if u.number_of_edges() else [{n} for n in u]
    communities=sorted(communities,key=lambda c: min(c))
    membership={n:i+1 for i,c in enumerate(communities) for n in c}
    bc=nx.betweenness_centrality(g, k=min(128,len(g)), seed=cfg['seed'], weight=None) if len(g)>1 else dict.fromkeys(g,0)
    metrics={n:{'inflow':float(g.in_degree(n,weight='weight')),'outflow':float(g.out_degree(n,weight='weight')),'in_degree':g.in_degree(n),'out_degree':g.out_degree(n),'betweenness':bc[n]} for n in g}
    maxflow=max([m['inflow']+m['outflow'] for m in metrics.values()]+[1]) or 1
    maxdegree=max([m['in_degree']+m['out_degree'] for m in metrics.values()]+[1])
    result=[]
    for row in nodes.sort_values('gid').to_dict('records'):
        n=row['gid']; m=metrics[n]; inc,out=m['inflow'],m['outflow']; di,do=m['in_degree'],m['out_degree']
        balanced=min(inc,out)/max(inc,out) if max(inc,out)>0 else 0
        role,score='peripheral',0.0
        if bc[n]>=cfg['coordinator_betweenness'] and di+do>=cfg['many_peers']: role,score='coordinator',min(1,bc[n]/(2*cfg['coordinator_betweenness']))
        elif di>=cfg['many_peers'] and di>do: role,score='consolidator',min(1,di/(2*cfg['many_peers']))
        elif do>=cfg['many_peers']: role,score='distributor',min(1,do/(2*cfg['many_peers']))
        elif not row['is_seed'] and row['depth']<4 and inc>0 and out>0 and balanced>=cfg['transit_ratio']: role,score='transit',balanced
        elif not row['is_seed'] and row['depth']<4 and inc>0 and out/inc<=cfg['terminal_ratio']: role,score='terminal',1-out/inc
        w=cfg['priority_weights']
        priority=w['flow']*(inc+out)/maxflow+w['degree']*(di+do)/maxdegree+w['betweenness']*bc[n]+w['seed']*int(row['is_seed'])
        evidence=f'Отправителей: {di}; получателей: {do}; вход: {inc:,.0f} ₸; выход: {out:,.0f} ₸; посредничество: {bc[n]:.3f}.'
        if row['is_seed']: evidence+=' Вход seed неполон.'
        if row['depth']>=4: evidence+=' Граница наблюдения.'
        result.append({**row,**m,'role':role,'role_score':round(score,4),'cluster_id':membership[n],'priority_score':round(min(1,max(0,priority)),4),'evidence':evidence[:200]})
    ranked=sorted(result,key=lambda n:(-n['priority_score'],n['gid']))
    clusters=[]
    for i,c in enumerate(communities,1):
        members=[n for n in ranked if n['gid'] in c]
        clusters.append({'cluster_id':i,'n_nodes':len(c),'n_seed':sum(n['is_seed'] for n in members),'sum_kzt_internal':sum(d['weight'] for a,b,d in g.edges(data=True) if a in c and b in c),'top_gids':'; '.join(n['gid'] for n in members[:5]),'hypothesis':'Гипотеза: связанная группа потоков; состав и роли требуют проверки.'})
    top=[{'rank':i,'gid':n['gid'],'role':n['role'],'priority_score':n['priority_score'],'why':n['evidence']} for i,n in enumerate(ranked[:max(20,min(50,len(ranked)))],1)]
    daily=tx.groupby(tx.date.dt.strftime('%Y-%m-%d')).sum_kzt.sum()
    return {'synthetic':synthetic,'nodes':result,'edges':[{'src':a,'dst':b,'sum_kzt':d['weight'],'n_tx':d['n_tx']} for a,b,d in g.edges(data=True)],'clusters':clusters,'top_nodes':top,'daily':[{'date':d,'sum_kzt':float(v)} for d,v in daily.items()],'summary':{'nodes':len(nodes),'edges':g.number_of_edges(),'transactions':len(tx),'volume':float(edges.sum_kzt.sum())}}

def write_outputs(data, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name,cols in COLUMNS.items():
        pd.DataFrame(data['nodes'] if name=='nodes_roles' else data[name],columns=cols).to_csv(output/f'{name}.csv',index=False,encoding='utf-8-sig')
    (output/'analysis.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')

def load(directory):
    return tuple(pd.read_parquet(Path(directory)/f'{name}.parquet') for name in ['nodes','edges','transactions'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='data'); p.add_argument('--output-dir',default='output'); p.add_argument('--demo',action='store_true'); args=p.parse_args()
    if args.demo:
        frames=demo(); Path(args.data_dir).mkdir(parents=True,exist_ok=True)
        for name,frame in zip(['nodes','edges','transactions'],frames): frame.to_parquet(Path(args.data_dir)/f'{name}.parquet',index=False)
    else: frames=load(args.data_dir)
    write_outputs(analyze(*frames,synthetic=args.demo),args.output_dir)
    print('Готово:',args.output_dir)

if __name__=='__main__': main()
