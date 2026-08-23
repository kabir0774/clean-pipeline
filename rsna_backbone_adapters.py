"""Auditable backbone adapters for the genuine RSNA pipeline.

These adapters expose encoders only. They do not load fold heads, predictions,
calibration constants, or test-derived artifacts.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import torch
import torch.nn as nn
from torchvision.models import resnet50


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def load_radimagenet_resnet50(checkpoint: str, expected_sha256: str | None = None) -> nn.Module:
    """Load a full ResNet-50 encoder checkpoint, refusing ambiguous artifacts.

    The checkpoint must contain a backbone state dict compatible with torchvision
    ResNet-50. A head-only/fold-trained bundle is rejected rather than silently
    used as a new architecture.
    """
    p=Path(checkpoint)
    if not p.is_file(): raise FileNotFoundError(p)
    if expected_sha256 and sha256(p).lower()!=expected_sha256.lower():
        raise ValueError('RadImageNet checkpoint SHA-256 mismatch')
    payload=torch.load(p,map_location='cpu',weights_only=True)
    state=payload.get('state_dict',payload) if isinstance(payload,dict) else payload
    if not isinstance(state,dict): raise ValueError('checkpoint is not a state dict')
    state={str(k).removeprefix('module.').removeprefix('backbone.'):v for k,v in state.items()}
    # Some legitimate RadImageNet exports store torchvision children in a
    # Sequential backbone: 0=conv1, 1=bn1, 4..7=layer1..layer4, 8=avgpool.
    child_map={'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4','8':'avgpool'}
    remapped={}
    for k,v in state.items():
        head=k.split('.',1)[0]
        if head in child_map:
            tail=k[len(head):].lstrip('.')
            remapped[child_map[head] + ('.'+tail if tail else '')]=v
        else:
            remapped[k]=v
    state=remapped
    model=resnet50(weights=None)
    # Encoder-only use: discard classifier if present; require convolutional keys.
    state={k:v for k,v in state.items() if not k.startswith('fc.')}
    missing,unexpected=model.load_state_dict(state,strict=False)
    if len(missing)>12 or not any(k.startswith('layer4.') for k in state):
        raise ValueError(f'checkpoint is not a compatible full ResNet-50 encoder; missing={len(missing)} unexpected={len(unexpected)}')
    return nn.Sequential(*list(model.children())[:-1])


def load_dinov2(source: str, train_last_blocks: int = 4) -> tuple[nn.Module,int]:
    """Load a local/public DINOv2 transformer and return encoder plus feature dim."""
    from transformers import AutoModel
    model=AutoModel.from_pretrained(source,local_files_only=Path(source).exists())
    for p in model.parameters(): p.requires_grad=False
    blocks=getattr(getattr(model,'encoder',None),'layer',None)
    if blocks is None: raise ValueError('DINOv2 source lacks encoder.layer; verify model contract')
    for block in blocks[-train_last_blocks:]:
        for p in block.parameters(): p.requires_grad=True
    if hasattr(model,'layernorm'):
        for p in model.layernorm.parameters(): p.requires_grad=True
    return model,int(model.config.hidden_size)
