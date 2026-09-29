"""Summarize completed, immutable experiments and render scientific figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/chain_v1'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#326fa8';GREEN='#1b8a79';ORANGE='#d99423';RED='#bd4642'

def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def main():
    m=pd.read_csv(RESULT/'metrics.csv',dtype={'fold':str})
    summary=m.groupby(['task','method','modality'],sort=False).agg(
        n_metric_rows=('macro_f1','size'),n_overlapping_folds=('fold','nunique'),
        file_macro_f1_mean=('macro_f1','mean'),file_macro_f1_min=('macro_f1','min'),
        file_macro_f1_max=('macro_f1','max'),file_accuracy_mean=('accuracy','mean')).reset_index()
    summary.to_csv(RESULT/'method_summary.csv',index=False)
    e=pd.read_csv(RESULT/'external_predictions.csv',dtype={'fold':str})
    eh=e[e.state=='bigNormal'].copy()
    eh['healthy_accepted']=eh.prediction.eq(0)
    ex=eh.groupby(['task','method','modality'],sort=False).agg(
        n_healthy_recordings=('healthy_accepted','size'),healthy_accepted=('healthy_accepted','sum'),
        mean_healthy_probability=('p0','mean')).reset_index()
    ex['false_alarm_fraction']=1-ex.healthy_accepted/ex.n_healthy_recordings
    ex.to_csv(RESULT/'external_summary.csv',index=False)

    fig,ax=plt.subplots(figsize=(13,5.7));ax.axis('off');ax.set(xlim=(0,13),ylim=(0,5.5))
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',
            facecolor=color,edgecolor='white',alpha=.15))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color='#203447')
    box(.15,3.2,2.6,1.6,'Contact: 4 kHz XYZ\n1 s / complete packet\nSigned DCT + fixed refs\nFrozen BearLLM FCN',BLUE)
    box(3.4,3.2,2.65,1.6,'128-D contact features\nSOURCE-only scaler\nAdapted 4-type head\nFrozen during distillation',BLUE)
    box(.15,.6,2.6,1.6,'Radar: 2 kHz in frame\n192 chirps / 100 ms\nFrame spectral shape\n5-800 Hz, 128 bands',GREEN)
    box(3.4,.6,2.65,1.6,'Radar MLP encoder\n128 -> 128 -> 64 -> 128\nRecording-bag losses\nNo fine time alignment',GREEN)
    box(7.0,1.9,2.4,1.6,'Frozen adapted head\nFour-type probabilities\nRadar-only inference\nROI validity status',ORANGE)
    box(10.05,1.9,2.55,1.6,'Explicit p4 -> p10 bridge\nFrozen linear3 + Qwen\nActual text generation\nConsistency check',BLUE)
    def arrow(a,b,**kw):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.7,color='#53677b',**kw))
    arrow((2.8,4),(3.25,4));arrow((2.8,1.4),(3.25,1.4))
    arrow((4.72,3.1),(4.72,2.3),linestyle='--')
    ax.text(4.95,2.68,'train only',fontsize=9,va='center')
    arrow((6.15,1.4),(7.1,2.3));arrow((6.15,4),(7.1,3.05));arrow((9.45,2.7),(9.95,2.7))
    ax.text(6.5,5.15,'Implemented chain and its evidence boundary',ha='center',fontsize=15,fontweight='bold')
    ax.text(6.5,.05,'Outer/keep ROI is inconsistent with target geometry. External motor transfer fails.\nBag matching is not synchronized-window matching.',ha='center',color=RED,fontsize=10)
    save(fig,'01_implemented_chain')

    names=['contact_original','contact_hidden','radar_frame_shape','radar_confound_raw_amp',
           'radar_confound_distance','supervised','head_only','distill','no_contrast','no_radar_CE','distill_phase','shuffled']
    labels=['Original contact head','Adapted contact head','Radar spectral + linear','Raw-amplitude control',
            'Distance-only control','Radar supervised MLP','Frozen head + CE','Full distillation','Without contrast',
            'Without radar CE*','With phase features','Wrong teacher bags']
    fig,axes=plt.subplots(1,2,figsize=(13,6.4),sharey=True)
    for ax,task,title in zip(axes,['four_class_provisional_roi','three_class_geometry_compatible'],
                             ['4 types: provisional ROI','3 types: outer ROI excluded']):
        sub=summary[summary.task==task].set_index('method').loc[names]
        mu=sub.file_macro_f1_mean.to_numpy();lo=sub.file_macro_f1_min.to_numpy();hi=sub.file_macro_f1_max.to_numpy()
        y=np.arange(len(names));colors=[RED,BLUE,GREEN,ORANGE,ORANGE,GREEN,BLUE,BLUE,BLUE,BLUE,BLUE,RED]
        ax.barh(y,mu,color=colors,alpha=.84,height=.67)
        ax.errorbar(mu,y,xerr=np.stack([mu-lo,hi-mu]),fmt='none',ecolor='#333',capsize=3,lw=1)
        for yy,val in zip(y,mu):ax.text(val+.018,yy,f'{val:.3f}',va='center',fontsize=9)
        ax.set_yticks(y,labels);ax.set_xlim(0,1.16);ax.set_xticks([0,.25,.5,.75,1]);ax.set_title(title)
        ax.set_xlabel('File macro-F1');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Recording holdouts reach a ceiling; distillation adds no measured classification gain',fontsize=13)
    fig.text(.5,-.015,'Bars: means. Whiskers: min-max across overlapping file partitions and optimization seeds, NOT confidence intervals.\n*Class labels still define positive teacher bags; this is not a fully unlabeled radar experiment.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'02_file_holdout_results')

    align=pd.read_csv(RESULT/'heldout_alignment.csv',dtype={'fold':str})
    a=align[(align.task=='four_class_provisional_roi')&align.method.isin(['head_only','distill','shuffled'])]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,(col,title) in enumerate([('mse','Contact-feature MSE (lower is better)'),('paired_cosine','Paired-bag cosine (higher is better)')]):
        ax=axes[j]
        for i,fold in enumerate(['0000','1111']):
            vals=a[a.fold==fold].set_index('method').loc[['head_only','distill','shuffled'],col]
            ax.bar(np.arange(3)+(i-.5)*.34,vals,width=.32,color=[BLUE,GREEN][i],label=['115200 -> 460800','460800 -> 115200'][i])
        ax.set_xticks(np.arange(3),['Frozen head + CE','Distillation','Wrong teacher'])
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Feature alignment improves; this does not establish precise time alignment',fontsize=12)
    fig.text(.5,-.025,'Seed 42. One test recording per class per direction: retrieval can reflect class matching alone.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'03_bag_alignment')

    methods=['contact_hidden','radar_frame_shape','supervised','head_only','distill']
    methodnames=['Adapted contact','Radar spectrum + linear','Radar supervised','Frozen head + CE','Radar distillation']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,bag in zip(axes,sorted(eh.bag_id.unique())):
        sub=eh[(eh.task=='four_class_provisional_roi')&(eh.bag_id==bag)].set_index('method').loc[methods]
        vals=sub[['p0','p1','p2','p3']].to_numpy()
        im=ax.imshow(vals,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for y in range(vals.shape[0]):
            for x in range(4):ax.text(x,y,f'{vals[y,x]:.3f}' if vals[y,x]>=.001 else '<.001',ha='center',va='center',fontsize=9,color='white' if vals[y,x]>.6 else '#172d3f')
        ax.set_xticks(range(4),['Normal','Inner','Outer','Ball']);ax.set_yticks(range(5),methodnames)
        ax.set_title('bigNormal '+bag.rsplit('-',1)[-1]+' (true: normal)',fontsize=10)
    fig.suptitle('External healthy motor: all five primary methods give false alarms (0/2 accepted)',fontsize=12)
    fig.text(.5,-.015,'Probabilities from all-known fits. Two recordings of one motor are case checks, not a population estimate.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'04_external_healthy_failures')

    verify=json.loads((RESULT/'verification.json').read_text())
    report=dict(independent_known_recordings=8,known_contact_windows=58,known_radar_windows=412,
        trained_student_instances=verify['trained_student_instances'],saved_model_reload_checks=verify['model_reload_count'],
        metric_rows=len(m),statistical_caveat='Repeated partitions are dependent. No independent-bearing confidence interval is estimated.',
        external_three_class_contact_spectral_exception=ex[(ex.task=='three_class_geometry_compatible')&(ex.method=='contact_spectral')].to_dict('records'))
    (RESULT/'report_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(summary.to_string(index=False))
    print(ex.to_string(index=False))

if __name__=='__main__':main()
