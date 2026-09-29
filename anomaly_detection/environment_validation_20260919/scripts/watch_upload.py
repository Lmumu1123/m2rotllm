"""Wait for a quiet upload directory; process a stable snapshot and watch for changes.

Quiet time is not proof of source-side completeness. By default an explicit
completion marker is also required, because paused transfers may look stable.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, time, traceback, hashlib
from datetime import datetime, timezone
from radar_environment import snapshot, ROOT, DATA


def stamp():return datetime.now(timezone.utc).isoformat()


def write_state(status,**extra):
    state=dict(timestamp_utc=stamp(),status=status,pid=os.getpid(),**extra)
    temp=ROOT/'results/.watch_status.tmp'
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2))
    temp.replace(ROOT/'results/watch_status.json')
    print(json.dumps(state,ensure_ascii=False),flush=True)


def run(args):
    previous=None;changed=time.monotonic();finished=None;started=time.monotonic()
    while time.monotonic()-started<args.max_hours*3600:
        current=snapshot()
        if current!=previous:
            changed=time.monotonic();previous=current
        quiet=time.monotonic()-changed
        gb=sum(x['size'] for x in current)/1e9
        marker=ROOT/'UPLOAD_COMPLETE.json'
        confirmed=marker.exists() or args.allow_unconfirmed_snapshot
        if current==finished:
            write_state('complete_for_current_snapshot',files=len(current),GB=gb,
                        quiet_seconds=round(quiet),report=str(ROOT/'环境泛化验证报告.md'))
            if args.stop_after_success:return
        elif quiet<args.quiet_seconds or len(current)<args.min_files or not confirmed:
            write_state('waiting_for_upload',files=len(current),GB=gb,
                        quiet_seconds=round(quiet),required_quiet_seconds=args.quiet_seconds,
                        completion_signal_received=marker.exists(),
                        completion_marker=str(marker),
                        previous_report_stale=finished is not None)
        else:
            write_state('processing_stable_snapshot',files=len(current),GB=gb,
                        quiet_seconds=round(quiet),completion_criterion='completion signal plus directory silence' if marker.exists() else 'unconfirmed stable snapshot')
            try:
                env=os.environ.copy()
                env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                subprocess.run([sys.executable,str(ROOT/'scripts/radar_environment.py'),'extract','--conjugate'],
                               check=True,env=env)
                if snapshot()!=current:raise RuntimeError('Upload resumed after extraction')
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_environment.py')],check=True,env=env)
                # Streaming hashes do not duplicate the dataset or modify raw files.
                checksums=[]
                for f in sorted(DATA.glob('*.bin')):
                    h=hashlib.sha256()
                    with f.open('rb') as reader:
                        for block in iter(lambda:reader.read(4*1024*1024),b''):h.update(block)
                    checksums.append(dict(file=f.name,sha256=h.hexdigest(),bytes=f.stat().st_size))
                if snapshot()!=current:raise RuntimeError('Upload resumed during evaluation/hash calculation')
                (ROOT/'results/input_sha256.json').write_text(json.dumps(checksums,indent=2))
                finished=current
                write_state('complete_for_current_snapshot',files=len(current),GB=gb,
                            report=str(ROOT/'环境泛化验证报告.md'))
                if args.stop_after_success:return
            except Exception as e:
                write_state('processing_failed_or_input_changed',error=str(e))
                traceback.print_exc()
                changed=time.monotonic()
        time.sleep(args.poll_seconds)
    write_state('monitor_time_limit_reached',max_hours=args.max_hours,
                completed_snapshot_exists=finished is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--quiet-seconds',type=int,default=300)
    p.add_argument('--poll-seconds',type=int,default=20)
    p.add_argument('--min-files',type=int,default=42)
    p.add_argument('--max-hours',type=float,default=12)
    p.add_argument('--stop-after-success',action='store_true')
    p.add_argument('--allow-unconfirmed-snapshot',action='store_true',
                   help='Explicit exploratory override; not source-side upload completion')
    run(p.parse_args())
