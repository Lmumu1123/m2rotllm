"""Encode one independently acquired, same-condition XYZ query/reference pair.

Inputs are .npy arrays of shape [4000,3], already checked for contiguous samples
within one packet. This command checks shape/numerics, not acquisition truth.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from contact_pipeline import prepare_pair
from teacher_bridge import CHECKPOINTS,load_teacher,embed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--query-run-id',required=True)
    p.add_argument('--reference-run-id',required=True)
    p.add_argument('--condition-id',required=True,help='Shared device/bearing geometry/rpm/load/mount/axis convention')
    p.add_argument('--teacher',choices=list(CHECKPOINTS),default='retrained_fcn')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    query=np.load(a.query,allow_pickle=False);ref=np.load(a.reference,allow_pickle=False)
    x=prepare_pair(query,ref,a.query_run_id,a.reference_run_id)
    torch.set_num_threads(2)
    model=load_teacher(a.teacher,a.device)
    fmap,h,logits=embed(model,torch.as_tensor(x,device=a.device))
    p0=torch.softmax(logits,dim=1)
    normalized=torch.nn.functional.normalize(h,dim=1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,feature_map=fmap.cpu().numpy(),hidden=h.cpu().numpy(),
        hidden_l2=normalized.cpu().numpy(),logits=logits.cpu().numpy(),probabilities=p0.cpu().numpy(),
        axis_order=np.array(['x','y','z']),available_band_mask=np.arange(24000)<4000)
    a.output.with_suffix('.json').write_text(json.dumps(dict(teacher=a.teacher,
        query=str(a.query.resolve()),reference=str(a.reference.resolve()),condition_id=a.condition_id,
        query_run_id=a.query_run_id,reference_run_id=a.reference_run_id,sample_rate_hz=4000,
        samples=4000,feature_map=list(fmap.shape),hidden=list(h.shape),logits=list(logits.shape),
        teacher_mode='frozen eval; BatchNorm does not update',
        limitation='Interface output only; local fault validity and independent reference provenance require separate verification'),ensure_ascii=False,indent=2))
    print('Saved',a.output,'shapes',tuple(fmap.shape),tuple(h.shape),tuple(logits.shape))


if __name__=='__main__':main()
