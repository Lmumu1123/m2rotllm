"""CPU-only engineering microbenchmark of an already trained radar student.

No labels are read. Includes feature standardization, encoder, frozen adapted
contact head and softmax; explicitly excludes acquisition and radar preprocessing.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import sys
import time
import numpy as np
import torch
from torch import nn
from run_chain import RadarEncoder
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

class InferencePipeline(nn.Module):

    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = RadarEncoder(int(checkpoint['input_dim']))
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        self.head = nn.Linear(128, len(checkpoint['classes']))
        self.head.weight.data.copy_(torch.as_tensor(checkpoint['contact_coef'], dtype=torch.float32))
        self.head.bias.data.copy_(torch.as_tensor(checkpoint['contact_intercept'], dtype=torch.float32))
        self.register_buffer('mean', torch.as_tensor(checkpoint['radar_mean'], dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(checkpoint['radar_scale'], dtype=torch.float32))
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.head(self.encoder((x - self.mean) / self.scale)).softmax(-1)

def timed(fn, inputs, warmup, iterations):
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    elapsed = []
    for i in range(iterations):
        x = inputs[i % len(inputs)]
        before = time.perf_counter_ns()
        fn(x)
        after = time.perf_counter_ns()
        elapsed.append((after - before) / 1000000.0)
    a = np.asarray(elapsed)
    return dict(iterations=iterations, warmup_iterations=warmup, p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)), mean_ms=float(a.mean()), min_ms=float(a.min()), max_ms=float(a.max()), samples_ms=elapsed)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'results/chain_v1/models/four_class_provisional_roi/all_known/distill_seed42.pt')
    p.add_argument('--features', type=Path, default=ROOT / 'radar/features.npz')
    p.add_argument('--output-prefix', type=Path, default=ROOT / 'results/latency_benchmark')
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--iterations', type=int, default=500)
    args = p.parse_args()
    if '/envs/m2vllm/' not in sys.executable:
        raise ValueError('Use conda environment m2vllm')
    if args.warmup < 200 or args.iterations < 500:
        raise ValueError('Require at least 200 warmup and 500 measured iterations')
    for suffix in ['.json', '.md']:
        if args.output_prefix.with_suffix(suffix).exists():
            raise ValueError('Refusing to overwrite an existing benchmark')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model_digest = sha(args.checkpoint)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck['method'] != 'distill' or ck['output_head_type'] != 'frozen_adapted_contact_head':
        raise ValueError('Expected distill model with frozen adapted contact head')
    with np.load(args.features, allow_pickle=False) as z:
        x = np.ascontiguousarray(z['frame_shape'], dtype=np.float32)
    if x.shape[1] != ck['input_dim'] or not np.isfinite(x).all():
        raise ValueError('Feature mismatch')
    model = InferencePipeline(ck)
    xt = torch.from_numpy(x)
    assert all((p.device.type == 'cpu' for p in model.parameters()))
    classes = np.asarray(ck['classes'], dtype=int)

    def combined(feature):
        sample = torch.from_numpy(np.asarray(feature, dtype=np.float32)).reshape(1, -1)
        prob = model(sample)[0].numpy()
        idx = int(np.argmax(prob))
        return dict(predicted_class=int(classes[idx]), confidence=float(prob[idx]), probabilities=prob.tolist())
    with torch.inference_mode():
        reference = x[:32].copy()
        reference -= ck['radar_mean']
        reference /= ck['radar_scale']
        pref = model.head(model.encoder(torch.from_numpy(reference))).softmax(-1)
        pdiff = float((pref - model(xt[:32])).abs().max())
        if pdiff > 1e-05:
            raise ValueError(f'Scaler precision changed probabilities: {pdiff}')
        tensors = [xt[i:i + 1] for i in range(len(x))]
        single = timed(model, tensors, args.warmup, args.iterations)
        merged = timed(combined, x, args.warmup, args.iterations)
        batches = [xt[i:i + 32] for i in range(0, len(xt) - 31, 32)]
        batch = timed(model, batches, args.warmup, args.iterations)
    batch.update(batch_size=32, windows_per_second_from_mean=32000 / batch['mean_ms'], amortized_ms_per_window=batch['mean_ms'] / 32)
    cpu = 'unknown'
    for line in Path(_c2r_resolve_path('/proc/cpuinfo')).read_text().splitlines():
        if line.startswith('model name'):
            cpu = line.split(':', 1)[1].strip()
            break
    params = dict(encoder=sum((p.numel() for p in model.encoder.parameters())), frozen_adapted_head=sum((p.numel() for p in model.head.parameters())), combined=sum((p.numel() for p in model.parameters())), trainable_during_inference=0, standardizer_buffer_values=model.mean.numel() + model.scale.numel())
    assert sha(args.checkpoint) == model_digest
    out = dict(status='complete', measurement='preloaded-feature CPU inference microbenchmark', utc_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checkpoint=str(args.checkpoint), checkpoint_sha256=model_digest, checkpoint_bytes=args.checkpoint.stat().st_size, parameters=params, feature_file=str(args.features), feature_key='frame_shape', input_shape_single=[1, int(ck['input_dim'])], software=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(), script_sha256=sha(Path(_c2r_resolve_path(__file__)))), hardware=dict(cpu_model=cpu, logical_cpus_host=os.cpu_count(), cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(), device='cpu', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=torch.cuda.is_initialized(), cpu_affinity_pinned=False, other_host_jobs_controlled=False, load_average=list(os.getloadavg())), protocol=dict(timer='time.perf_counter_ns', evaluation_mode=True, torch_inference_mode=True, dropout_disabled=True, gc_enabled=gc.isenabled(), labels_read_or_used=False, features_and_weights_preloaded=True, precision='float32 model and standardization buffers', preprocessing_scaler_probability_max_abs_diff=pdiff, timing_includes='standardization + radar encoder + frozen adapted contact head + softmax', combined_adds='NumPy feature to tensor view + argmax/class mapping + confidence/probability result dict', excluded=['2-second data acquisition', 'ADC parsing', 'range FFT/ROI selection', 'radar feature extraction', 'file IO', 'model load', 'contact encoder', 'LLM generation'], interpretation='engineering cost only; not an end-to-end real-time or fault-accuracy claim'), single_preloaded_tensor=single, single_feature_to_result=merged, batch32=batch)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    text = f"# 雷达学生 CPU 推理成本\n\n测量对象是已训练的 `four_class_provisional_roi/all_known/distill_seed42.pt`，采用 128 维已提取雷达特征、学生 encoder 和冻结的**适配后接触式分类头**。只测工程成本，不读取或使用标签。\n\n| 项目 | 实测值 |\n|---|---:|\n| Encoder 参数量 | {params['encoder']:,} |\n| 冻结适配分类头参数量 | {params['frozen_adapted_head']:,} |\n| 总模型参数量 | {params['combined']:,} |\n| 标准化缓冲值数量 | {params['standardizer_buffer_values']} |\n| 保存 checkpoint 文件大小 | {out['checkpoint_bytes']:,} 字节（{out['checkpoint_bytes'] / 1024:.2f} KiB） |\n\n| 测量范围 | p50（ms） | p95（ms） | 均值（ms） |\n|---|---:|---:|---:|\n| 单窗口预载 Tensor：标准化→encoder→冻结头→softmax | {single['p50_ms']:.4f} | {single['p95_ms']:.4f} | {single['mean_ms']:.4f} |\n| 单窗口内存特征→结果字典：包含上述模型过程及输入/结果转换 | {merged['p50_ms']:.4f} | {merged['p95_ms']:.4f} | {merged['mean_ms']:.4f} |\n| 32 窗口批次：预载 Tensor→softmax | {batch['p50_ms']:.4f} | {batch['p95_ms']:.4f} | {batch['mean_ms']:.4f} |\n\n每项分别预热 {args.warmup} 次，正式测量 {args.iterations} 次。单窗口每次仅输入 `[1,128]`，没有把全部 614 窗口的批吞吐冒充单条延迟。Batch 32 的平均吞吐为 {batch['windows_per_second_from_mean']:,.0f} 窗口/秒，摊销值 {batch['amortized_ms_per_window']:.5f} ms/窗口仅用于吞吐解释。\n\nCPU：`{cpu}`。PyTorch `{torch.__version__}`，NumPy `{np.__version__}`，Python `{platform.python_version()}`；conda `m2vllm`。Torch intra-op 固定 2 线程，inter-op 1 线程，模型为 float32、eval/inference_mode。GPU 被隐藏且未初始化，本测量不争用训练 GPU。CPU 未绑核，其他主机任务未隔离，因此这些数值是当时服务器实测微基准，不能当作确定性的硬实时保证。\n\n**不包含** ADC 解析、距离 FFT/选门、特征提取、2 秒采集、文件 I/O、权重加载、接触式 encoder 或 LLM 推理。两秒窗口的采集等待也不能从这些耗时中省略，因此不能称为完整链路端到端实时延迟。模型当前仍带有 provisional ROI 限制；成本低不证明诊断有效。\n\n输入标准化的 float32 工程实现与原 source scaler 算术结果的概率最大绝对差为 `{pdiff:.3g}`。完整逐次耗时、软件信息、模型与脚本 SHA256 见 [latency_benchmark.json](latency_benchmark.json)。\n\n复现时使用新输出名，避免覆盖本次证据：\n\n```bash\n/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \\\n  python {Path(_c2r_resolve_path(__file__)).resolve()} \\\n  --output-prefix {ROOT / 'results/latency_benchmark_repeat'}\n```\n"
    args.output_prefix.with_suffix('.md').write_text(text)
    print(json.dumps(dict(parameters=params, checkpoint_bytes=out['checkpoint_bytes'], single_tensor_p50_ms=single['p50_ms'], single_tensor_p95_ms=single['p95_ms'], feature_to_result_p50_ms=merged['p50_ms'], feature_to_result_p95_ms=merged['p95_ms'], batch32_windows_per_second=batch['windows_per_second_from_mean'], output=str(args.output_prefix)), indent=2))
if __name__ == '__main__':
    main()
