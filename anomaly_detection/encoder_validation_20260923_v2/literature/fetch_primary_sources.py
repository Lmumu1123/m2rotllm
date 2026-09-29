from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,requests
import re,html
ROOT=Path(__file__).resolve().parent
sources={
'functional_stitching':'https://proceedings.mlr.press/v267/smith25a.html',
'cka':'https://proceedings.mlr.press/v97/kornblith19a.html',
'probe_controls':'https://aclanthology.org/D19-1275/',
'rkd':'https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html',
'biox':'https://arxiv.org/html/2510.02276v2',
'pagkd':'https://arxiv.org/html/2601.09209v1',
'ucmkd':'https://arxiv.org/html/2606.10504v1',
'resi':'https://proceedings.iclr.cc/paper_files/paper/2025/hash/2eef1f75516b0cfd3e944345e5f88c08-Abstract-Conference.html',
'cka_reliability':'https://arxiv.org/abs/2210.16156',
'unifault':'https://arxiv.org/html/2504.01373v2',
'biox_official':'https://github.com/chenqi-li/BioX-Bridge',
'pagkd_official':'https://github.com/Huster-Hq/PaGKD',
'ucmkd_official':'https://github.com/anhducchu/UCMKD',
}
def one(it):
 key,url=it
 try:
  r=requests.get(url,timeout=40)
  clean=re.sub(r'<(script|style|nav|header|footer)\b.*?</\1>', '', r.text, flags=re.S|re.I)
  txt=html.unescape(re.sub('<[^>]+>', '\n', clean))
  txt='\n'.join(x.strip() for x in txt.splitlines() if x.strip())
  title=re.search('<title>(.*?)</title>',r.text,re.S|re.I)
  (ROOT/'sources').mkdir(exist_ok=True)
  (ROOT/'sources'/f'{key}.txt').write_text(txt)
  return dict(key=key,url=url,final_url=r.url,status=r.status_code,sha256=hashlib.sha256(r.content).hexdigest(),fetched_utc=datetime.now(timezone.utc).isoformat(),title=html.unescape(title.group(1)) if title else '',characters=len(txt),text_file=f'sources/{key}.txt')
 except Exception as e:return dict(key=key,url=url,error=str(e))
with ThreadPoolExecutor(max_workers=8) as ex:
 results=list(ex.map(one,sources.items()))
(ROOT/'source_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
for r in results:print(r)
