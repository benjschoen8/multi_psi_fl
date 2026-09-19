"""Offline text-only ablation; run from the repository root:

HF_HUB_OFFLINE=1 python3 tests/benchmark_keyword_attention.py

Uses one client per language, the unchanged anchor bank, tau=.05 and LADDER.
No images, classifier training, threshold search or ground-truth weighting.
The old sentence/four-head/pooled path is reconstructed explicitly.
"""
import ast, json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from sentence_transformers import SentenceTransformer
from label_mapping.rt_protocol import TEXT_ANCHORS, run_rt_protocol, pair_metrics
ns={'np':np}
tree=ast.parse(Path('test_mnist_split_new.py').read_text())
keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('describe','encode_keywords') or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LANGS' for t in n.targets)]
exec(compile(ast.Module(body=keep,type_ignores=[]),'<helpers>','exec'),ns)
m=SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', local_files_only=True,device='cpu')
encode=lambda texts:m.encode(texts,normalize_embeddings=True,show_progress_bar=False)
K=encode(TEXT_ANCHORS)
results=[]
for split_name, splits in [('default',[[0,1,2,3,4],[5,6,7,8],[0,1,5,6,9]]),('all_digits',[list(range(10))]*3)]:
    for variant in ['sentence_pooled_4heads','keywords_alpha_uniform','keywords_alpha_idf','identity_alpha',
                    'keywords_log_whiten_confident']:
        clients={}
        templates=['the handwritten digit {}','手寫數字{}','el dígito escrito a mano {}']
        for i,digits in enumerate(splits):
            names=[ns['LANGS'][i][1][d] for d in digits]
            c={'summ':dict.fromkeys(range(len(digits)),np.zeros(1)), 'count':dict.fromkeys(range(len(digits)),20)}
            if variant=='sentence_pooled_4heads':
                c['desc_vecs']=dict(enumerate(encode([templates[i].format(n) for n in names])))
            else:
                descriptions=[ns['describe'](n,'short' if variant=='identity_alpha' else 'keywords') for n in names]
                c['keyword_vecs']=dict(enumerate(ns['encode_keywords'](encode,descriptions)))
                if variant in ('keywords_alpha_idf', 'keywords_log_whiten_confident'):
                    c['keywords']=dict(enumerate(descriptions))
            clients[i]=c
        revised = variant == 'keywords_log_whiten_confident'
        table,edges,diag=run_rt_protocol(clients,None,method='attn',psi='plain',K_text=K,
            coord='log_whiten' if revised else 'pooled' if variant=='sentence_pooled_4heads' else 'alpha',
            attn_confidence='top' if revised else 'positive',
            n_heads=4 if variant=='sentence_pooled_4heads' else 1,log=lambda *_:None)
        metrics=pair_metrics(table,lambda a,b:splits[a[0]][a[1]]==splits[b[0]][b[1]])
        row={'split':split_name,'variant':variant,**metrics}
        results.append(row)
        print(json.dumps(row),flush=True)
Path('logs').mkdir(exist_ok=True)
Path('logs/keyword_attention_revision_validation.json').write_text(json.dumps(results,indent=2))
