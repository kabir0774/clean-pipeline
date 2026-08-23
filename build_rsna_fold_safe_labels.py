#!/usr/bin/env python3
"""Create fold-specific weak labels for RSNA Knee without global V4 blend leakage.

For outer fold k, per-target V2/GPT weights are selected only using the gold
studies not in fold k. Fold-k gold studies are excluded from training. All other
studies receive a soft V2/GPT target and confidence weight.

This is intentionally NOT a recreation of the globally selected V4 blend.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def auc(y, score):
    y=np.asarray(y, dtype=int); score=np.asarray(score, dtype=float)
    n1=(y==1).sum(); n0=(y==0).sum()
    if not n1 or not n0: return np.nan
    rank=pd.Series(score).rank(method='average').to_numpy()
    return float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def ranks(x): return pd.Series(x).rank(method='average', pct=True).to_numpy()

def make_folds(gold, n_folds, seed):
    """Deterministic grouped multilabel folds with hard capacities.

    The 58 gold studies are the evaluation ruler. A tiny fold is not acceptable:
    capacity is fixed first (e.g. 12,12,12,11,11), then labels are balanced.
    """
    y=gold[TARGETS].to_numpy(int); n=len(gold); totals=y.sum(0)
    if (totals < n_folds).any():
        bad=[TARGETS[i] for i,v in enumerate(totals) if v<n_folds]
        raise ValueError(f'not enough positives for {n_folds} folds: {bad}')
    base, extra=divmod(n,n_folds); capacity=np.array([base+int(k<extra) for k in range(n_folds)])
    rng=np.random.default_rng(seed); rarity=1/np.maximum(totals,1)
    order=sorted(range(n),key=lambda i: (-(y[i]*rarity).sum(), -y[i].sum(), rng.random()))
    counts=np.zeros((n_folds,len(TARGETS))); sizes=np.zeros(n_folds,int); target=totals/n_folds; f=np.full(n,-1,int)
    for i in order:
        available=np.flatnonzero(sizes<capacity)
        costs=[]
        for k in available:
            c=counts.copy(); c[k]+=y[i]
            costs.append((((c-target)**2)/np.maximum(target,1)).sum())
        k=int(available[int(np.argmin(costs))]); f[i]=k; counts[k]+=y[i]; sizes[k]+=1
    if sorted(sizes.tolist()) != sorted(capacity.tolist()): raise RuntimeError(f'fold capacity failure: {sizes} vs {capacity}')
    for k in range(n_folds):
        if np.any(np.unique(y[f==k],axis=0).shape[0] < 1): raise RuntimeError(f'empty fold {k}')
        for j,t in enumerate(TARGETS):
            if len(np.unique(y[f==k,j])) < 2: raise RuntimeError(f'fold {k} has single-class target {t}')
    gold=gold.copy(); gold['outer_fold']=f
    return gold

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-csv',required=True); p.add_argument('--v2-csv',required=True); p.add_argument('--gpt-csv',required=True); p.add_argument('--out',required=True)
    p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=20260822); p.add_argument('--grid-steps',type=int,default=21)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(a.train_csv,dtype={'StudyInstanceUID':str}); v2=pd.read_csv(a.v2_csv,dtype={'StudyInstanceUID':str}); gpt=pd.read_csv(a.gpt_csv,dtype={'StudyInstanceUID':str})
    for name,d in [('train',train),('v2',v2),('gpt',gpt)]:
        missing=[c for c in ['StudyInstanceUID',*TARGETS] if c not in d]
        if missing: raise ValueError(f'{name} missing {missing}')
        if d.StudyInstanceUID.duplicated().any(): raise ValueError(f'{name} has duplicate study IDs')
    if set(train.StudyInstanceUID)!=set(v2.StudyInstanceUID) or set(train.StudyInstanceUID)!=set(gpt.StudyInstanceUID):
        raise ValueError('train/V2/GPT study IDs must match exactly')
    gold=train.loc[train[TARGETS].notna().all(axis=1),['StudyInstanceUID',*TARGETS]].copy()
    gold[TARGETS]=gold[TARGETS].astype(int)
    if len(gold)<a.folds: raise ValueError('insufficient complete gold rows')
    gold=make_folds(gold,a.folds,a.seed); gold.to_csv(out/'gold_outer_folds.csv',index=False)
    source=v2[['StudyInstanceUID',*TARGETS]].merge(gpt[['StudyInstanceUID',*TARGETS]],on='StudyInstanceUID',suffixes=('_v2','_gpt'),validate='one_to_one')
    gold_for_merge=gold.rename(columns={t:f'{t}_gold' for t in TARGETS})
    source=source.merge(gold_for_merge[['StudyInstanceUID','outer_fold',*[f'{t}_gold' for t in TARGETS]]],on='StudyInstanceUID',how='left',validate='one_to_one')
    grid=np.linspace(0,1,a.grid_steps)
    manifest={'folds':a.folds,'seed':a.seed,'targets':TARGETS,'folds_detail':{}}
    for k in range(a.folds):
        # Only non-held-out gold examples select this fold's target-specific blend.
        select=source[source.outer_fold.notna() & (source.outer_fold!=k)]
        weights={}; selection_auc={}
        for t in TARGETS:
            y=select[f'{t}_gold'].to_numpy(int); x=ranks(select[f'{t}_v2']); z=ranks(select[f'{t}_gpt'])
            options=[(auc(y,(1-w)*x+w*z),float(w)) for w in grid]
            score,w=max(options,key=lambda q:-np.inf if np.isnan(q[0]) else q[0])
            weights[t]=w; selection_auc[t]=score
        fold_df=pd.DataFrame({'StudyInstanceUID':source.StudyInstanceUID})
        is_gold=source.outer_fold.notna().to_numpy(); is_holdout=(source.outer_fold==k).to_numpy()
        fold_df['is_gold']=is_gold.astype(int); fold_df['train_enabled']=(~is_holdout).astype(int); fold_df['outer_fold']=source.outer_fold
        for t in TARGETS:
            w=weights[t]; soft=(1-w)*source[f'{t}_v2'].to_numpy(float)+w*source[f'{t}_gpt'].to_numpy(float)
            # Gold labels replace weak labels except the held-out records are disabled from training.
            gold_value=source[f'{t}_gold'].to_numpy(float); soft=np.where(is_gold,gold_value,soft)
            # Model-generated score near 0.5 means no report evidence. Gold labels always carry full weight.
            conf=np.clip(2*np.abs(soft-0.5),0,1); conf=np.where(is_gold,1.0,conf)
            fold_df[t]=soft; fold_df[f'{t}__confidence']=conf
        fold_df.to_csv(out/f'labels_fold_{k}.csv',index=False)
        manifest['folds_detail'][str(k)]={'weights_v2_gpt':weights,'selection_auc_train_gold':selection_auc,'n_train_enabled':int((~is_holdout).sum()),'n_gold_holdout':int(is_holdout.sum())}
    (out/'label_builder_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'gold_rows':len(gold),'folds':a.folds,'out':str(out)},indent=2))

if __name__=='__main__': main()
