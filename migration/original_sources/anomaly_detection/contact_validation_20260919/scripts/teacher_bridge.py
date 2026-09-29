"""Frozen BearLLM classification representations; never load the full LLM."""
from pathlib import Path
import sys
import torch

REPO=Path('/home/huangyating/BearLLM')
sys.path.insert(0,str(REPO))
from models.FCN import FaultClassificationNetwork
from scripts.evaluate_mbhm import load_adapter

ASSETS=Path('/media/nas_users/huangyating/bearllm-assets')
RUN=Path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42')
CHECKPOINTS={
    'retrained_fcn':RUN/'pretrain/fcn',
    'official_final_adapter':ASSETS/'bearllm_weights',
    'retrained_final_adapter':RUN/'finetune/weights',
}


def load_teacher(name,device='cpu'):
    if name=='retrained_fcn':
        model=FaultClassificationNetwork()
        model.encoder.load_state_dict(torch.load(CHECKPOINTS[name]/'feature_encoder.pth',map_location='cpu',weights_only=True))
        model.classifier.load_state_dict(torch.load(CHECKPOINTS[name]/'classifier.pth',map_location='cpu',weights_only=True))
        model=model.to(device).eval()
    else:
        model=load_adapter(CHECKPOINTS[name],device,include_lora=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval() # Critical: freeze BatchNorm running buffers as well as gradients.
    return model


def embed(model,x):
    """Input [B,2,24000]; map [B,128,47], h [B,128], logits [B,10]."""
    if x.ndim!=3 or tuple(x.shape[1:])!=(2,24000):
        raise ValueError('Expected [batch, query/reference, 24000], not XYZ channels')
    with torch.inference_mode():
        if hasattr(model,'encoder'):
            feature=model.encoder(x);head=model.classifier
        else:
            feature=model.feature_encoder(x);head=model.alignment_layer
        hidden=torch.relu(head.linear1(feature.reshape(len(x),-1)))
        logits=head.linear2(hidden)
        if not all(torch.isfinite(v).all() for v in (feature,hidden,logits)):
            raise ValueError('Nonfinite teacher outputs')
    return feature,hidden,logits
