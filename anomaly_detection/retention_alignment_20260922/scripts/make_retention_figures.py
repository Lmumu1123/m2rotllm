"""Scientific figures from measured source-retention and alignment results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    s=pd.read_csv(ROOT/'results/retained_head_conservative/source_group_metrics.csv')
    fig,ax=plt.subplots(figsize=(11,4.5));names=['ALL','CWRU','DIRG','HIT','IMS','JUST','MFPT','NCEPU','PU','XJTU']
    for j,(v,color,label) in enumerate([('v0','#247d92','AC query branch'),('query_no_demean','#c99136','DC-preserving query branch')]):
        d=s[(s.split=='test')&(s.variant==v)].set_index('source').loc[names]
        ax.bar(np.arange(len(names))+(j-.5)*.34,d.delta_acc4_pp,width=.33,label=label,color=color)
    ax.axhline(-.5,color='#b74747',linestyle='--',label='Allowed drop: 0.5 percentage points')
    ax.axhline(0,color='#888',lw=.7);ax.set_ylim(-.57,.15);ax.set_xticks(np.arange(len(names)),names)
    ax.set_ylabel('Change in four-type accuracy (percentage points)')
    ax.set_title('Conservative shared head: original-data regression by source dataset')
    ax.legend(loc='lower right',fontsize=9)
    fig.text(.5,-.035,'Fixed original query/reference protocol; 12,279 test queries. Small conditions and unseen motors require separate checks.',ha='center',fontsize=9)
    save(fig,'01_source_retention')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    a=pd.read_csv(ROOT/'radar/stage_a_summary.csv');a=a[a.task=='four_class_provisional_roi'].set_index('method')
    b=pd.read_csv(ROOT/'radar/stage_b_v0/metrics.csv');b=b[b.role=='test'].groupby('method').mean(numeric_only=True)
    names=['Old head:\nembedding only','Old head:\nCE + alignment','Retained head:\nfull alignment']
    vals=[[a.loc['embedding_only','macro_f1'],a.loc['ce_feat_1','macro_f1'],b.loc['ce_feat_1_kd','macro_f1']],
          [a.loc['embedding_only','MSE'],a.loc['ce_feat_1','MSE'],b.loc['ce_feat_1_kd','mse_standardized']]]
    for ax,v,title in zip(axes,vals,['File macro-F1 (higher is better)','Standardized embedding MSE (lower is better)']):
        ax.bar(range(3),v,color=['#ac5252','#ce9b40','#247d92']);ax.set_xticks(range(3),names,fontsize=9);ax.set_title(title,fontsize=11)
        for i,y in enumerate(v):ax.text(i,y+.025,f'{y:.3f}',ha='center',fontsize=10)
        ax.set_ylim(0,max(v)*1.17);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Correcting the contact head resolves the measured alignment/diagnosis conflict',fontsize=12)
    fig.text(.5,-.07,'AC query branch; two recording holdout directions x three seeds. Only eight unique main recordings.\nOuter-fault radar ROI is still inconsistent with geometry: four-type scores are provisional.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'02_alignment_and_diagnosis')

if __name__=='__main__':main()
