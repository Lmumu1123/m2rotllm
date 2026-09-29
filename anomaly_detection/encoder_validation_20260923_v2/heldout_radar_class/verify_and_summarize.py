"""Recompute decisions from saved hidden+original head; audit exclusions."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax,logsumexp
from sklearn.preprocessing import StandardScaler
import run_heldout as run

def main():
    out=run.ROOT/'results';rm=pd.read_csv(out/'radar_metadata.csv');cm=pd.read_csv(out/'contact_metadata.csv')
    x=np.load(run.RADAR/'features.npz')['frame_shape'];h=np.load(run.CONTACT/'contact_features.npz')['hidden_mean']
    splits=json.loads((out/'splits.json').read_text());met=pd.read_csv(out/'metrics.csv');files=pd.read_csv(out/'file_predictions.csv')
    assert len(splits)==8 and len(list((out/'models').glob('*.pt')))==96
    checks=[];contact=[]
    for sp in splits:
        fold=sp['fold'];held=sp['heldout_label'];ctr=cm.bag_id.isin(sp['student_train_bags']);rtr=rm.bag_id.isin(sp['student_train_bags'])
        assert (cm.loc[ctr,'label']!=held).all() and (rm.loc[rtr,'label']!=held).all()
        assert sp['contact_scaler_fit_rows']==np.flatnonzero(ctr).tolist() and sp['radar_scaler_fit_rows']==np.flatnonzero(rtr).tolist()
        cs=StandardScaler().fit(h[ctr],sample_weight=run.api.weights(cm[ctr]));rs=StandardScaler().fit(x[rtr],sample_weight=run.api.weights(rm[rtr]))
        orig=np.load(sp['shared_head']);w=orig['weight10'];b=orig['bias10']
        # Head's existing contact skill for this withheld radar class.
        logits=h@w.T+b;p=softmax(np.stack([logsumexp(logits[:,g],axis=1) for g in run.api.GROUPS],1),axis=1)
        for role,group in [('prior_contact_train',sp['head_prior_contact_training_bags']),('contact_test',sp['test_bags'])]:
            take=cm.bag_id.isin(group)&cm.label.eq(held);pp=p[take]
            contact.append(dict(fold=fold,heldout_label=held,role=role,n_windows=int(take.sum()),window_correct=int((pp.argmax(1)==held).sum()),
                window_accuracy=float((pp.argmax(1)==held).mean()),recording_correct=int(pp.mean(0).argmax()==held),recording_probability=float(pp.mean(0)[held])))
        for method in run.METHODS:
            for seed in run.SEEDS:
                key=f'{fold}__heldout{held}__{method}__seed{seed}'
                z=np.load(out/'windows'/f'{key}.npz');ck=torch.load(out/'models'/f'{key}.pt',weights_only=True,map_location='cpu')
                assert np.array_equal(ck['head_weight'].numpy(),w) and np.array_equal(ck['head_bias'].numpy(),b)
                assert np.allclose(ck['contact_mean'].numpy(),cs.mean_,rtol=0,atol=1e-12)
                assert np.allclose(ck['contact_scale'].numpy(),cs.scale_,rtol=0,atol=1e-12)
                assert np.allclose(ck['radar_mean'].numpy(),rs.mean_,rtol=0,atol=1e-12)
                assert np.allclose(ck['radar_scale'].numpy(),rs.scale_,rtol=0,atol=1e-12)
                selected=rm.iloc[z['source_metadata_row']]
                assert set(selected.bag_id)==set(sp['test_bags']) and np.array_equal(selected.label,z['label'])
                logits=z['hidden']@w.T+b;pp=softmax(np.stack([logsumexp(logits[:,g],axis=1) for g in run.api.GROUPS],1),axis=1)
                error=float(abs(pp-z['probabilities4']).max());assert error<2e-5
                for scope,take in [('all_four',np.ones(len(pp),bool)),('seen_three',z['label']!=held),('unseen_one',z['label']==held)]:
                    row=met[(met.fold==fold)&(met.heldout_label==held)&(met.method==method)&(met.seed==seed)&(met.unit=='window')&(met.scope==scope)].iloc[0]
                    assert int(row.correct)==int((pp[take].argmax(1)==z['label'][take]).sum())
                    assert int(row.predicted_heldout)==int((pp[take].argmax(1)==held).sum())
                for bag in sp['test_bags']:
                    take=z['bag_id']==bag;pred=int(pp[take].mean(0).argmax())
                    row=files[(files.fold==fold)&(files.heldout_label==held)&(files.method==method)&(files.seed==seed)&(files.bag_id==bag)].iloc[0]
                    assert int(row.prediction)==pred
                checks.append(dict(model=key,probabilities_recomputed_from_original_head_max_error=error,
                    exclusion_and_scaler_recomputed=True,reported_window_and_recording_counts_match=True))
    pd.DataFrame(contact).to_csv(out/'contact_prior_class_skill.csv',index=False)
    run.savej(out/'independent_verification.json',dict(models=checks,models_verified=96,teacher_head_unchanged=True,
        heldout_radar_excluded_from_scalers_and_train=True,contact_scaler_also_excludes_heldout_class=True))
    summarize(out)
    print(pd.read_csv(out/'summary.csv').query("scope=='unseen_one'").to_string(index=False))

def summarize(out):
    summary=pd.read_csv(out/'summary.csv');byclass=pd.read_csv(out/'summary_by_heldout_class.csv');prior=pd.read_csv(out/'contact_prior_class_skill.csv')
    names=dict(ce_only='只用类别',matched_bag_mse='真实接触均值',full='完整方法',label_code_mse='人为类别向量')
    lines=['# 完全不训练某类雷达时，还能判断该类吗？','',
        '本轮结果：四种方法对完全没有参与雷达训练的第四类，窗口和录制召回均为0%；见过的三类仍接近100%。现有训练方案没有展示从三类雷达拓展到第四类的能力。','',
        '这项实验检查已训练的跨模态关系能否拓展到没有用于雷达训练的一类；不是普通的四分类精度重复，也不是整个系统没有见过这一故障。','',
        '## 实验规则','',
        '每次拿走一类故障的全部雷达数据。训练只用另外三类的三条录制，雷达和接触特征标准化也只使用这三类。测试另一次录制的四类数据，分别看见过的三类与从未训练的第四类。四输出接触分类头保持固定，允许它已从接触数据学过第四类。','',
        '两个留出方向、四种被拿走的类别、四种训练方法、三个固定种子，共96个模型。400步结束即评估，没有按结果挑类别、种子或训练步数。使用修正距离后的雷达数据。','',
        '## 实际结果','',
        '“未见类召回”是该类有多少被认对；“误报未见类”是见过的三类有多少被错判为第四类。百分比按保存的答对计数汇总；每种方法的未见类测试总共涉及412个不同雷达窗口、8次实际录制，各重复三个种子，得到1236次窗口判定和24次录制判定。种子不增加物理样本量，窗口也不等于独立轴承样本。','',
        '| 方法 | 见过三类窗口准确率 | 未见类窗口召回 | 未见类录制召回 | 见过三类误报未见类（窗口） |','|---|---:|---:|---:|---:|']
    for m in run.METHODS:
        ss=summary[(summary.method==m)&(summary.scope=='seen_three')&(summary.unit=='window')].iloc[0]
        uw=summary[(summary.method==m)&(summary.scope=='unseen_one')&(summary.unit=='window')].iloc[0]
        ur=summary[(summary.method==m)&(summary.scope=='unseen_one')&(summary.unit=='recording')].iloc[0]
        lines.append(f'| {names[m]} | {ss.pooled_accuracy:.2%} | {uw.pooled_accuracy:.2%} | {ur.pooled_accuracy:.2%} | {ss.pooled_predicted_heldout_rate:.2%} |')
    lines+=['','| 被完全排除的雷达类别 | 只用类别 | 真实接触均值 | 完整方法 | 人为类别向量 |','|---|---:|---:|---:|---:|']
    for held,label in enumerate(['正常','内圈','外圈','滚动体']):
        vals=[]
        for m in run.METHODS:
            d=byclass[(byclass.heldout_label==held)&(byclass.method==m)&(byclass.scope=='unseen_one')&(byclass.unit=='window')].iloc[0]
            vals.append(f'{d.accuracy:.2%}')
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines+=['','上表为未见类的窗口召回，全部类别都报告。每个格子只有两个实际录制，种子重复不构成独立物理证据。','',
        '## 解释边界','',
        '- 原分类头保留第四类输出，因此失败不是因为删除了它。同一隐藏特征→分类头读取路径，对真实接触测试录制是8/8判断正确；窗口为46/58。contact_prior_class_skill.csv 保留逐类详细结果，窗口与录制口径不混淆。',
        '- 如果见过三类仍然很好、第四类却失败，说明当前训练方案没有展示把既有接触知识自动拓展到未训练雷达类别的能力。普通四类高分不能替代这一验证。',
        '- 这不证明所有跨模态方法都不可能拓展，也不等于当前系统没有实用价值；它明确了现有训练数据和目标的适用范围。',
        '- 真实接触均值方法每类仅有一个目标，学习目标仍然接近三个类别代表点。要支持丰富知识，需要增加同类多条件录制、连续物理属性任务或独立严重度任务，而不能仅增加四分类重复种子。',
        '- 原分类头之前使用过四类接触训练数据，所以这不是整个系统的零样本故障识别；所有组共享同一个条件，比较的是雷达侧迁移。','',
        '## 记录','',
        '- protocol_before_results.json：运行前固定协议。',
        '- results/splits.json：每个类别每个方向的真正训练录制、两种Scaler拟合行、原头先前接触训练录制。',
        '- results/windows 与 models：全部96个模型、测试隐藏特征及四类概率。',
        '- results/metrics.csv、file_predictions.csv、summary_by_heldout_class.csv：逐种子、逐类、逐录制计数。',
        '- results/independent_verification.json：从保存hidden与原分类头重新计算概率和判定，重算三类Scaler并核对排除条件。',
        '- 首次启动在任何完整模型评估前遇到 PyTorch inference-mode 目标张量反传限制；改为 no_grad 生成固定监督目标后从头运行。同一协议未改变，启动错误输出保存在 results_setup_error 和 setup_error.log。','']
    (run.ROOT/'留一雷达类别实验_结果与解释.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
