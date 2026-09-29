"""Build report figures, portable reading copies, and evidence manifest; no training."""
from pathlib import Path
import base64
import csv
import hashlib
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.render_dependencies'))
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from lxml import etree
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

BASE = Path('/home/huangyating/anomaly_detection')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def records(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def savefig(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT/'figures'/f'{name}.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def figure_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set(xlim=(0,14),ylim=(0,6.5)); ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=.1',
                                   facecolor=color,edgecolor='none'))
        ax.text(x+w/2,y+h-.28,title,ha='center',va='center',weight='bold',color='#173047',fontsize=11)
        ax.text(x+w/2,y+.4,body,ha='center',va='center',fontsize=10,color='#243746',linespacing=1.45)
    def arrow(x1,y1,x2,y2,dashed=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,
                                    linewidth=1.6,color='#52697c',linestyle='--' if dashed else '-'))
    ax.text(.1,6.15,'Implemented contact-to-radar diagnosis pipeline',fontsize=18,weight='bold',color='#173047')
    ax.text(.1,5.76,'Solid: implemented   |   Dashed: supervision or proposed validation   |   No event-level alignment claim',fontsize=10,color='#52697c')
    box(.15,3.85,2.3,1.35,'Contact XYZ','4 kHz; complete packets\n1 s DCN + reference','#dcebf7')
    box(3.05,3.85,2.65,1.35,'Frozen contact encoder','BearLLM FCN + linear1\n128-D ReLU hidden','#dcebf7')
    box(.15,1.7,2.3,1.35,'Radar complex IQ','Range ROI + valid frames\n128-band spectral shape','#dcefe9')
    box(3.05,1.7,2.65,1.35,'Trainable radar encoder','MLP: 128 → 128 → 64 → 128\nCompatible ReLU hidden','#dcefe9')
    box(7.0,2.78,2.45,1.4,'One shared head','Retained 10-way logits\nExact 4-class grouping','#f8e6c5')
    box(10.25,2.78,3.05,1.4,'Diagnosis + language','Native p10 → original projection\nFrozen Qwen + existing LoRA','#e8e6f5')
    arrow(2.52,4.52,2.95,4.52); arrow(2.52,2.37,2.95,2.37)
    arrow(5.78,4.52,6.91,3.78); arrow(5.78,2.37,6.91,3.16); arrow(9.53,3.48,10.15,3.48)
    arrow(4.36,3.76,4.36,3.15,True)
    ax.text(5.02,3.38,'Bag MSE / KD\n+ contrast',fontsize=9,va='center',color='#52697c')
    ax.text(7,2.0,'Head adaptation: old-data replay + KD; encoder frozen.\nRadar training: head frozen; true-label CE also used.',fontsize=10,color='#52697c')
    ax.add_patch(FancyBboxPatch((.15,.2),13.15,.93,boxstyle='round,pad=.06',facecolor='#faf8f4',
                               edgecolor='#b18045',linestyle='--',linewidth=1.3))
    ax.text(6.72,.66,'NOT YET VALIDATED: corrected outer-fault ROI · reliable cross-motor teacher · independent-domain transfer gains\nDynamic/event correspondence · calibrated rejection · end-to-end deployment cost',
            ha='center',va='center',fontsize=10,color='#875c2d',linespacing=1.6)
    savefig(fig,'01_pipeline')

def figure_results(totals):
    by={(r['modality'],r['method']):r for r in totals}
    contact=by['contact','shared_contact_head']; radar=by['radar','ce_feat_1_kd']
    fig,axes=plt.subplots(1,3,figsize=(14,5.4),gridspec_kw={'width_ratios':[1,1.3,.85]})
    colors=['#3d779e','#ca8a35']
    ax=axes[0]
    vals=[100*float(contact['accuracy']),100*float(radar['accuracy'])]
    ax.bar([0,1],vals,color=colors,width=.62)
    for i,(v,r) in enumerate(zip(vals,[contact,radar])):
        ax.text(i,v+1.8,f'{v:.2f}%\n{r["correct"]}/{r["n"]}',ha='center',fontsize=10)
    ax.set(ylim=(0,115),yticks=[0,25,50,75,100],ylabel='Window accuracy (%)',xticks=[0,1],
           xticklabels=['Contact\n1-second windows','Radar*\n2-second windows'],title='A. Two-fold recording holdout')
    ax=axes[1]
    rr=[by['radar',m] for m in ['embedding_only','ce_only','ce_feat_1_kd']]
    vv=[100*float(r['accuracy']) for r in rr]
    ax.bar(range(3),vv,color=['#92b4aa','#7396ad','#ca8a35'],width=.65)
    for i,(v,r) in enumerate(zip(vv,rr)):
        ax.text(i,v+1.6,f'{v:.2f}%\n{r["correct"]}/{r["n"]}',ha='center',fontsize=10)
    ax.set(ylim=(0,115),yticks=[0,25,50,75,100],xticks=range(3),xticklabels=['Bag MSE','CE only','Full loss'],
           title='B. Same head; radar ablations*')
    ax=axes[2]
    ax.bar([0,1],[0,0],color=colors)
    ax.scatter([0,1],[0,0],marker='x',s=100,color='#a7463e',clip_on=False,zorder=5)
    for i in [0,1]: ax.text(i,6,'0/2',ha='center',color='#a7463e',fontsize=14,weight='bold')
    ax.set(xlim=(-.6,1.6),ylim=(0,115),yticks=[0,25,50,75,100],xticks=[0,1],xticklabels=['Contact','Radar'],
           title='C. Other motor: recording level')
    ax.text(.5,56,'Conservative all-known model\nOnly normal recognition tested\nMeasurement quality unresolved',
            ha='center',fontsize=9,color='#52697c',linespacing=1.6)
    for ax in axes: ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('High development scores do not establish independent physical generalization',fontsize=15,weight='bold',y=1.03)
    fig.text(.01,-.045,'* Radar includes 100 outer-fault windows from an invalid ROI. All three methods classify 8/8 recordings correctly.\nOnly eight main recordings; same bearing per class. Panel C is a different model and evaluation scope.',fontsize=10,color='#52697c')
    fig.tight_layout(w_pad=2)
    savefig(fig,'02_results')

def set_eastasia(style, name='宋体'):
    style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),name)

def inline(par, node, bold=False, italic=False):
    if node.text:
        r=par.add_run(node.text);r.bold=bold;r.italic=italic
    for ch in node:
        if ch.tag=='a':
            link=OxmlElement('w:hyperlink');link.set(qn('r:id'),par.part.relate_to(ch.get('href',''),RT.HYPERLINK,is_external=True))
            run=OxmlElement('w:r');rp=OxmlElement('w:rPr');co=OxmlElement('w:color');co.set(qn('w:val'),'23668A');rp.append(co)
            run.append(rp);t=OxmlElement('w:t');t.text=''.join(ch.itertext());run.append(t);link.append(run);par._p.append(link)
        elif ch.tag=='br':par.add_run().add_break()
        elif ch.tag=='img':
            path=Path(ch.get('src',''))
            if path.exists():par.add_run().add_picture(str(path),width=Inches(6.5))
        else:
            inline(par,ch,bold or ch.tag in ('strong','b'),italic or ch.tag in ('em','i'))
        if ch.tail:par.add_run(ch.tail)

def make_docx(body, dst):
    doc=Document()
    sec=doc.sections[0];sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
    sec.left_margin=sec.right_margin=Inches(.78);sec.top_margin=sec.bottom_margin=Inches(.7)
    normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10.5);set_eastasia(normal)
    normal.paragraph_format.space_after=Pt(5);normal.paragraph_format.line_spacing=1.14
    for name,size in [('Title',22),('Heading 1',16),('Heading 2',12),('Heading 3',11)]:
        st=doc.styles[name];st.font.name='Calibri';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string('173D56');set_eastasia(st,'黑体')
    sec.header.paragraphs[0].text='Contact-to-Radar Research  |  2026-09-23  |  Internal progress report'
    sec.header.paragraphs[0].style=doc.styles['Caption']
    foot=sec.footer.paragraphs[0];foot.alignment=2
    foot.add_run('Page ');fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
    root=etree.HTML(body).find('body')
    def block(node):
        if node.tag in ('h1','h2','h3','h4'):
            level=max(0,int(node.tag[1])-1);p=doc.add_heading(level=level);inline(p,node)
        elif node.tag=='p':
            p=doc.add_paragraph();inline(p,node)
        elif node.tag in ('ul','ol'):
            for li in node.findall('li'):
                p=doc.add_paragraph(style='List Bullet' if node.tag=='ul' else 'List Number');inline(p,li)
        elif node.tag=='pre':
            p=doc.add_paragraph();p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(7)
            r=p.add_run(''.join(node.itertext()).rstrip());r.font.name='Consolas';r.font.size=Pt(9)
            set_eastasia(r,'宋体')
        elif node.tag=='table':
            rows=node.findall('.//tr');cols=max(len(list(r)) for r in rows)
            table=doc.add_table(rows=0,cols=cols);table.style='Light Shading Accent 1'
            for i,tr in enumerate(rows):
                cells=table.add_row().cells
                for j,td in enumerate(tr):
                    inline(cells[j].paragraphs[0],td,bold=(i==0))
                    for p in cells[j].paragraphs:
                        p.paragraph_format.space_after=Pt(3)
                        for run in p.runs:run.font.size=Pt(9)
                if i==0:
                    repeat=OxmlElement('w:tblHeader');table.rows[-1]._tr.get_or_add_trPr().append(repeat)
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
        elif node.tag=='blockquote':
            p=doc.add_paragraph(style='Quote');p.add_run(''.join(node.itertext()).strip())
        else:
            for ch in node:block(ch)
    for node in root:block(node)
    doc.core_properties.title=dst.stem;doc.core_properties.subject='Verified results, limitations, and SenSys research plan'
    doc.core_properties.comments='Generated from the accompanying Markdown. No new model training.'
    doc.save(dst)

def render(md):
    body=markdown.markdown(md.read_text(),extensions=['tables','fenced_code','sane_lists'])
    make_docx(body,md.with_suffix('.docx'))
    htmlbody=body
    for target in re.findall(r'<img[^>]+src="([^"]+)"',body):
        p=Path(target)
        if p.exists():htmlbody=htmlbody.replace('src="'+target+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
    css='''body{font-family:system-ui,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#243746;line-height:1.75;max-width:1080px;margin:40px auto;padding:0 28px}h1,h2,h3{color:#173d56;line-height:1.4}h1{font-size:28px}h2{margin-top:36px;border-bottom:2px solid #dcebf1;padding-bottom:8px}h3{margin-top:26px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #d6e0e6;padding:8px 10px;vertical-align:top}th{background:#edf4f8}tr:nth-child(even){background:#f8fafb}pre{background:#f1f5f8;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}code{font-family:Consolas,monospace}a{color:#226c92;text-decoration:none}img{max-width:100%;height:auto}strong{color:#183d55}blockquote{border-left:4px solid #ca8a35;padding-left:18px}@media print{body{margin:0;padding:0;font-size:10pt;line-height:1.55}h1{font-size:20pt}h2{font-size:15pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}table{font-size:8pt}tr,img,pre{break-inside:avoid}thead{display:table-header-group}a{color:inherit}@page{size:A4;margin:16mm}}'''
    md.with_suffix('.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+md.stem+'</title><style>'+css+'</style><body>'+htmlbody+'</body></html>')

def main():
    source_paths=[BASE/'acceptance_contact98_radar95/metrics.csv',
      BASE/'acceptance_contact98_radar95/student_teacher_audit/window_totals.csv',
      BASE/'contact_quality_alignment_audit_20260922/leakage/source_accuracy_retention.csv',
      BASE/'contact_quality_alignment_audit_20260922/leakage/source_duplicate_summary.json',
      BASE/'contact_quality_alignment_audit_20260922/leakage/source_train_eval_duplicate_counts.json',
      BASE/'contact_quality_alignment_audit_20260922/quality/verification.json',
      BASE/'contact_quality_alignment_audit_20260922/theory/verification.json']
    manifest=[]
    for i,p in enumerate(source_paths,1):
        out=ROOT/'evidence'/f'{i:02d}_{p.name}';shutil.copy2(p,out)
        manifest.append({'source':str(p),'snapshot':str(out.relative_to(ROOT)),'sha256':digest(p)})
    totals=records(source_paths[1]);metrics=records(source_paths[0])
    lookup={(r['modality'],r['method']):r for r in totals}
    assert (int(lookup['contact','shared_contact_head']['correct']),int(lookup['contact','shared_contact_head']['n']))==(46,58)
    assert (int(lookup['radar','ce_feat_1_kd']['correct']),int(lookup['radar','ce_feat_1_kd']['n']))==(411,412)
    for mode,method in [('contact','shared_contact_head'),('radar','ce_feat_1_kd')]:
        rows=[r for r in metrics if r['scope']=='heldout_recordings' and r['modality']==mode]
        assert sum(int(r['window_correct']) for r in rows)==int(lookup[mode,method]['correct'])
        assert sum(int(r['window_n']) for r in rows)==int(lookup[mode,method]['n'])
    retained=[r for r in records(source_paths[2]) if r['version']=='conservative' and r['variant']=='v0'][0]
    assert f"{100*float(retained['acc4']):.4f}"=='99.0472'
    figure_pipeline();figure_results(totals)
    docs=sorted(ROOT.glob('导师汇报*.md'))
    bad=[];nlinks=0
    for p in docs:
        for target in re.findall(r'\]\((/[^)]+)\)',p.read_text()):
            nlinks+=1
            if not Path(re.sub(r':\d+$','',target)).exists():bad.append((str(p),target))
        render(p)
    assert not bad,bad
    verification={'status':'passed','new_training':False,'new_accuracy_experiment':False,
       'source_inputs_unchanged':all(digest(p)==m['sha256'] for p,m in zip(source_paths,manifest)),
       'checked_local_links':nlinks,'missing_local_links':bad,'score_tables_cross_checked':True,
       'figures':['01_pipeline','02_results'],'source_manifest':manifest,
       'documents':[{ 'file':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in docs],
       'python':sys.executable}
    assert verification['source_inputs_unchanged']
    (ROOT/'evidence/verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k not in ['source_manifest','documents']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
