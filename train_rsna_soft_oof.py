#!/usr/bin/env python3
"""Train a genuine RSNA Knee image model with fold-specific soft labels.

No test metadata calibration, report-text inference, public-LB fitting, inherited
competition heads, or global V4 blend is used. Labels must come from
build_rsna_fold_safe_labels.py.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import (
    ResNet18_Weights, ConvNeXt_Tiny_Weights, EfficientNet_V2_S_Weights,
    resnet18, convnext_tiny, efficientnet_v2_s,
)
from rsna_knee_genuine import TARGETS, SLOTS, Config, KneeStudyDataset, macro_auc, weighted_bce, seed_everything
from rsna_backbone_adapters import load_dinov2, load_radimagenet_resnet50


def predict(model, loader, device):
    model.eval(); ids=[]; out=[]
    with torch.no_grad():
        for batch in loader:
            x=batch['image'].to(device,non_blocking=True); m=batch['valid'].to(device,non_blocking=True)
            with autocast(enabled=device.type=='cuda'): logits=model(x,m)
            out.append(torch.sigmoid(logits).float().cpu().numpy()); ids.extend(batch['study_id'])
    return ids, np.concatenate(out,axis=0)


class MultiArchStudyMIL(nn.Module):
    """2.5D slice encoder and target-query pooling across MRI slots/slices."""
    def __init__(self, architecture: str, pretrained: bool = True, dino_source: str | None = None, rad_checkpoint: str | None = None, rad_sha256: str | None = None, dino_train_last_blocks: int = 4):
        super().__init__()
        self.kind='cnn'
        if architecture in ('dinov2','dinov2_small'):
            if not dino_source: raise ValueError('--dino-source is required for dinov2')
            self.encoder, dim = load_dinov2(dino_source, train_last_blocks=dino_train_last_blocks)
            self.kind='dino'
        elif architecture == 'radimagenet_resnet50':
            if not rad_checkpoint: raise ValueError('--rad-checkpoint is required for radimagenet_resnet50')
            self.encoder=load_radimagenet_resnet50(rad_checkpoint, rad_sha256); dim=2048
        elif architecture == 'resnet18':
            net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder=nn.Sequential(*list(net.children())[:-1]); dim=512
        elif architecture == 'convnext_tiny':
            net=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder=nn.Sequential(net.features, net.avgpool); dim=768
        elif architecture == 'efficientnet_v2_s':
            net=efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder=nn.Sequential(net.features, net.avgpool); dim=1280
        else:
            raise ValueError(f'unknown architecture {architecture}; choose resnet18, convnext_tiny, efficientnet_v2_s')
        self.architecture=architecture
        self.slot=nn.Parameter(torch.randn(len(SLOTS),dim)*0.02)
        self.pos=nn.Parameter(torch.randn(64,dim)*0.02)
        self.query=nn.Parameter(torch.randn(len(TARGETS),dim)*0.02)
        self.attn=nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(0.2),nn.Linear(dim,len(TARGETS)))
        self.register_buffer('mean',torch.tensor([0.485,0.456,0.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([0.229,0.224,0.225])[None,:,None,None])

    def forward(self,image,valid):
        b,s,k,h,w=image.shape
        pad=torch.nn.functional.pad(image,(0,0,0,0,1,1),mode='replicate')
        tri=torch.stack([pad[:,:,i:i+k] for i in range(3)],dim=3).reshape(b*s*k,3,h,w)
        encoded=self.encoder((tri-self.mean)/self.std)
        if self.kind=='dino':
            feat=encoded.last_hidden_state[:,0]
        else:
            feat=encoded.flatten(1)
        feat=feat.reshape(b,s*k,-1)
        mask=valid.reshape(b,s*k).bool(); empty=~mask.any(1)
        if empty.any(): mask=mask.clone(); mask[empty,0]=True
        si=torch.arange(s,device=image.device).repeat_interleave(k); pi=torch.arange(k,device=image.device).repeat(s)
        feat=feat+self.slot[si][None]+self.pos[pi][None]
        q=self.query[None].expand(b,-1,-1)
        z,_=self.attn(q,feat,feat,key_padding_mask=~mask,need_weights=False)
        return self.head(z).diagonal(dim1=1,dim2=2)


def train_fold(k, train_df, val_df, series, root, cfg, architecture, out, dino_source=None, rad_checkpoint=None, rad_sha256=None, dino_train_last_blocks=4):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_ds=KneeStudyDataset(train_df,series,root,'train_series',cfg,train=True)
    va_ds=KneeStudyDataset(val_df,series,root,'train_series',cfg,train=False)
    dl=lambda ds,shuffle:DataLoader(ds,batch_size=cfg.batch_size,shuffle=shuffle,num_workers=cfg.workers,pin_memory=True,persistent_workers=cfg.workers>0)
    tr,va=dl(tr_ds,True),dl(va_ds,False)
    model=MultiArchStudyMIL(architecture,cfg.pretrained,dino_source=dino_source,rad_checkpoint=rad_checkpoint,rad_sha256=rad_sha256,dino_train_last_blocks=dino_train_last_blocks).to(device)
    if cfg.compile and hasattr(torch,'compile'): model=torch.compile(model)
    pos=train_df[TARGETS].sum().to_numpy(np.float32); neg=len(train_df)-pos
    pw=torch.tensor(np.clip(np.sqrt(neg/np.maximum(pos,1)),1,5),device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs); scaler=GradScaler(enabled=device.type=='cuda')
    best=-np.inf; bestp=None
    for epoch in range(cfg.epochs):
        model.train()
        for batch in tr:
            opt.zero_grad(set_to_none=True)
            x=batch['image'].to(device,non_blocking=True); m=batch['valid'].to(device,non_blocking=True)
            y=batch['target'].to(device,non_blocking=True); c=batch['confidence'].to(device,non_blocking=True)
            with autocast(enabled=device.type=='cuda'): loss=weighted_bce(model(x,m),y,c,pw)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        sched.step(); _,p=predict(model,va,device); score,_=macro_auc(val_df[TARGETS].to_numpy(np.int8),p)
        print(f'fold={k} arch={architecture} epoch={epoch+1}/{cfg.epochs} gold_val_macro_auc={score:.5f}')
        if score>best:
            best=score;bestp=p.copy()
            torch.save({'fold':k,'architecture':architecture,'state_dict':model.state_dict(),'score':score},out/f'{architecture}_fold_{k}.pt')
    return bestp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--labels-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--study-list',default=None,help='optional CSV with StudyInstanceUID for an isolated smoke subset')
    p.add_argument('--architecture',choices=['resnet18','convnext_tiny','efficientnet_v2_s','dinov2','dinov2_small','radimagenet_resnet50'],default='resnet18')
    p.add_argument('--dino-source','--dinov2-source',dest='dino_source',default=None,help='local DINOv2 model directory or approved model ID')
    p.add_argument('--dino-train-last-blocks','--dinov2-train-last-blocks',dest='dino_train_last_blocks',type=int,default=4)
    p.add_argument('--rad-checkpoint','--radimagenet-checkpoint',dest='rad_checkpoint',default=None,help='full RadImageNet ResNet-50 encoder checkpoint')
    p.add_argument('--rad-sha256','--radimagenet-sha256',dest='rad_sha256',default=None,help='required SHA-256 for the RadImageNet checkpoint')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=3);p.add_argument('--workers',type=int,default=6);p.add_argument('--image-size',type=int,default=256);p.add_argument('--slices-per-slot',type=int,default=12);p.add_argument('--slice-sampling',choices=['middle','full'],default='middle');p.add_argument('--middle-fraction',type=float,default=0.60);p.add_argument('--no-laterality-canonicalization',action='store_true');p.add_argument('--geometry-log',default='runs/geometry_fallbacks.jsonl');p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20260822);p.add_argument('--compile',action='store_true');p.add_argument('--no-pretrained',action='store_true')
    a=p.parse_args(); seed_everything(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=Config(root=a.root,labels_csv='',output=a.output,seed=a.seed,image_size=a.image_size,slices_per_slot=a.slices_per_slot,slice_sampling=a.slice_sampling,middle_fraction=a.middle_fraction,canonicalize_laterality=not a.no_laterality_canonicalization,geometry_log=a.geometry_log,batch_size=a.batch_size,workers=a.workers,epochs=a.epochs,lr=a.lr,weight_decay=a.weight_decay,pretrained=not a.no_pretrained,compile=a.compile)
    series=pd.read_csv(Path(a.root)/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    subset_ids=None
    if a.study_list:
        subset_ids=set(pd.read_csv(a.study_list,dtype={'StudyInstanceUID':str})['StudyInstanceUID'].astype(str))
        if len(subset_ids)==0: raise ValueError('--study-list is empty')
        print(f'Using isolated study subset: {len(subset_ids)}')
    oof_rows=[]
    for k in range(5):
        labels=pd.read_csv(Path(a.labels_dir)/f'labels_fold_{k}.csv',dtype={'StudyInstanceUID':str})
        if subset_ids is not None:
            labels=labels[labels.StudyInstanceUID.astype(str).isin(subset_ids)].reset_index(drop=True)
            gold_count=int(labels.is_gold.eq(1).sum())
            if gold_count != 58: raise ValueError(f'study subset must include all 58 gold studies; found {gold_count}')
        required=['StudyInstanceUID','is_gold','train_enabled','outer_fold',*TARGETS,*[f'{t}__confidence' for t in TARGETS]]
        miss=[c for c in required if c not in labels];
        if miss: raise ValueError(f'fold {k} labels missing {miss}')
        tr=labels[labels.train_enabled.eq(1)].reset_index(drop=True)
        va=labels[labels.is_gold.eq(1)&labels.outer_fold.eq(k)].reset_index(drop=True)
        if set(tr.StudyInstanceUID)&set(va.StudyInstanceUID): raise RuntimeError('held-out gold leakage')
        pred=train_fold(k,tr,va,series,Path(a.root),cfg,a.architecture,out,a.dino_source,a.rad_checkpoint,a.rad_sha256,a.dino_train_last_blocks)
        part=va[['StudyInstanceUID',*TARGETS]].copy()
        for j,t in enumerate(TARGETS):part[f'pred_{t}']=pred[:,j]
        oof_rows.append(part)
    oof=pd.concat(oof_rows,ignore_index=True); metric,per=macro_auc(oof[TARGETS].to_numpy(np.int8),oof[[f'pred_{t}' for t in TARGETS]].to_numpy())
    oof.to_csv(out/f'{a.architecture}_gold_oof.csv',index=False);(out/f'{a.architecture}_metrics.json').write_text(json.dumps({'macro_gold_oof_auc':metric,'per_target_auc':per,'architecture':a.architecture},indent=2))
    print(json.dumps({'macro_gold_oof_auc':metric,'per_target_auc':per},indent=2))

if __name__=='__main__':main()
