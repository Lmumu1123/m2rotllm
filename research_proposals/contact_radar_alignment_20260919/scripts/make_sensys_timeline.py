"""Schedule diagram only; not a claim of completed experiments."""
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
tasks=[
    ('Existing-data environment audit','2026-09-19','2026-09-22','#12877D'),
    ('Paired collection + contact teacher','2026-09-21','2026-09-26','#2875B8'),
    ('Frozen-head radar transfer + pilot tests','2026-09-25','2026-10-03','#2875B8'),
    ('Specimen / environment / mounting tests','2026-10-03','2026-10-13','#805BAA'),
    ('Strong baselines + mechanism ablations','2026-10-10','2026-10-21','#805BAA'),
    ('Freeze results + system measurements','2026-10-21','2026-10-27','#D78428'),
    ('Complete draft + abstract registration','2026-10-24','2026-10-30','#D78428'),
    ('Evidence audit + final submission','2026-10-30','2026-11-06','#BF4A50'),
]
fig,ax=plt.subplots(figsize=(14,6.4))
for i,(name,start,end,color) in enumerate(tasks):
    a=mdates.date2num(datetime.fromisoformat(start));b=mdates.date2num(datetime.fromisoformat(end))
    ax.barh(i,b-a,left=a,height=.6,color=color,alpha=.92)
ax.set_yticks(range(len(tasks)),[x[0] for x in tasks]);ax.invert_yaxis()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SA))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.set_xlim(datetime(2026,9,18),datetime(2026,11,7))
for day,label in [('2026-10-02','Pilot target'),('2026-10-29','Abstract (AoE)'),('2026-11-05','Paper (AoE)')]:
    x=datetime.fromisoformat(day);ax.axvline(x,color='#647589',ls='--',lw=1)
    ax.text(x,-.7,label,rotation=0,ha='center',va='bottom',fontsize=9,color='#24364B')
ax.grid(axis='x',alpha=.2)
ax.spines[['top','right','left']].set_visible(False)
ax.set_title('14-day end-to-end pilot and SenSys 2027 submission plan',loc='left',pad=34,fontsize=16,weight='bold')
fig.text(.02,.02,'Suggested schedule. Upload completion, paired data, and signal observability determine feasibility.',fontsize=10,color='#647589')
fig.tight_layout(rect=(0,.06,1,.96))
for ext in ['png','svg','pdf']:
    fig.savefig(ROOT/'figures'/f'07_sensys_timeline.{ext}',dpi=180,bbox_inches='tight')
print('Saved SenSys timeline diagram (PNG/SVG/PDF).')
