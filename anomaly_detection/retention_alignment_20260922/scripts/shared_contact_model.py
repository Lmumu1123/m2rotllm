"""Four-class public interface retaining a ten-logit internal knowledge branch.

Contact DCN and radar embeddings use exactly the same frozen diagnostic head.
No original checkpoint is edited. The radar encoder must be trained for this
specific 128-dimensional hidden space and head version.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ORIGINAL = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn'))
GROUPS = ((0,), (1, 2, 3), (7, 8, 9), (4, 5, 6))
CLASS_NAMES = ('normal', 'inner', 'outer', 'ball')

class SharedFourClassModel(nn.Module):

    def __init__(self, head_path, original_weights=ORIGINAL):
        super().__init__()
        spec = importlib.util.spec_from_file_location('frozen_bearllm_fcn', _c2r_resolve_path('/home/huangyating/BearLLM/models/FCN.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.contact_encoder = mod.FeatureEncoder()
        self.contact_encoder.load_state_dict(torch.load(original_weights / 'feature_encoder.pth', map_location='cpu', weights_only=True))
        old = torch.load(original_weights / 'classifier.pth', map_location='cpu', weights_only=True)
        self.contact_linear1 = nn.Linear(128 * 47, 128)
        self.contact_linear1.load_state_dict({k.removeprefix('linear1.'): v for k, v in old.items() if k.startswith('linear1.')})
        with np.load(head_path, allow_pickle=False) as ck:
            self.register_buffer('weight10', torch.tensor(ck['weight10'], dtype=torch.float32))
            self.register_buffer('bias10', torch.tensor(ck['bias10'], dtype=torch.float32))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def contact_hidden(self, dcn_query_reference):
        if dcn_query_reference.ndim != 3 or dcn_query_reference.shape[1:] != (2, 24000):
            raise ValueError('Expected [batch, query/reference=2, DCN=24000]')
        return F.relu(self.contact_linear1(self.contact_encoder(dcn_query_reference).flatten(1)))

    def classify_embedding(self, hidden):
        if hidden.ndim != 2 or hidden.shape[1] != 128:
            raise ValueError('Expected aligned [batch,128] hidden')
        logits10 = F.linear(hidden, self.weight10, self.bias10)
        logits4 = torch.stack([torch.logsumexp(logits10[:, g], dim=-1) for g in GROUPS], -1)
        return dict(logits4=logits4, probabilities4=logits4.softmax(-1), logits10=logits10, probabilities10=logits10.softmax(-1), hidden=hidden)

    def forward(self, dcn_query_reference):
        """Normal classification interface returns FOUR logits."""
        return self.classify_embedding(self.contact_hidden(dcn_query_reference))['logits4']

    def contact_views(self, views, *, pool):
        """Pool explicitly: local teacher uses hidden; MBHM uses probabilities.

        views: [queries, views_or_references, 2, 24000]. The two pooling rules
        do not commute with softmax and must never be silently interchanged.
        """
        if views.ndim != 4 or views.shape[2:] != (2, 24000):
            raise ValueError('Expected [N,V,2,24000]')
        n, v = views.shape[:2]
        hidden = self.contact_hidden(views.reshape(n * v, 2, 24000)).reshape(n, v, 128)
        if pool == 'hidden':
            return self.classify_embedding(hidden.mean(1))
        if pool != 'probability':
            raise ValueError('Specify hidden or probability pooling')
        out = self.classify_embedding(hidden.reshape(n * v, 128))
        p10 = out['probabilities10'].reshape(n, v, 10).mean(1)
        p4 = torch.stack([p10[:, g].sum(-1) for g in GROUPS], -1)
        return dict(probabilities4=p4, probabilities10=p10, hidden=hidden.mean(1))
