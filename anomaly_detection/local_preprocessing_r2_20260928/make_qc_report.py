#!/usr/bin/env python3
"""Create a small self-contained HTML QC report; reads exported H5, never raw BIN."""
from pathlib import Path
import argparse
import html
import json
import h5py
import numpy as np


def chart(path):
    with h5py.File(path, "r") as f:
        r = f["pilot_range_m"][:]
        p = f["pilot_positive_dynamic_power"][:]
        q = json.loads(f.attrs["qc_json"])
        lo, hi = q["physical_roi_bounds_m"]
        center = q["selected_center_range_m"]
    maxrange = max(1.2, hi + .1)
    keep = r <= maxrange
    r, p = r[keep], p[keep]
    # Per-chart normalization is for display only; exported signal is unchanged.
    y = np.maximum(-70, 10 * np.log10(np.maximum(p, 1e-30) / max(float(p.max()), 1e-30)))
    xx = lambda v: 48 + float(v) / maxrange * 470
    yy = lambda v: 155 - (float(v) + 70) / 70 * 125
    points = " ".join(f"{xx(a):.2f},{yy(b):.2f}" for a, b in zip(r, y))
    ticks = "".join(f'<text x="{xx(x):.1f}" y="176" text-anchor="middle">{x:.1f}</text>'
                    for x in np.linspace(0, maxrange, 7))
    return f'''<svg viewBox="0 0 550 205" role="img" aria-label="距离动态能量轮廓">
    <rect x="{xx(lo):.2f}" y="30" width="{xx(hi)-xx(lo):.2f}" height="125" fill="#d9eee7"/>
    <line x1="48" y1="155" x2="518" y2="155" stroke="#777"/>
    <line x1="48" y1="30" x2="48" y2="155" stroke="#777"/>
    <polyline points="{points}" fill="none" stroke="#1965b0" stroke-width="1.6"/>
    <line x1="{xx(center):.2f}" x2="{xx(center):.2f}" y1="30" y2="155" stroke="#dd632d" stroke-dasharray="5 3"/>
    <text x="5" y="33">0 dB</text><text x="0" y="155">−70 dB</text>{ticks}
    <text x="225" y="199">距离 / m</text></svg>''', q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="preprocess_local process output directory")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    paths = sorted((root / "upload" / "radar").glob("*/recording.json"))
    if not paths:
        parser.error("No completed radar exports")
    cards = []
    contacts = {}
    for path in paths:
        rec = json.loads(path.read_text(encoding="utf-8"))
        svg, q = chart(path.parent / "radar.h5")
        contact = rec["contact"]
        contacts[rec["contact_id"]] = contact
        flags = q["quality_flags"]
        text = "; ".join(flags) if flags else "未触发已实现的质量提示；仍需核对真实距离与采集日志"
        cards.append(f'''<article><h3>{html.escape(rec['recording_id'])}</h3>
          <p>标记距离 {q['declared_distance_cm']:g} cm；选取峰 {q['selected_center_range_m']:.3f} m；
          时长 {q['nominal_export_duration_s']:.2f} s；{q['frames_exported']} 帧。</p>{svg}
          <p class="{'warn' if flags else 'note'}">{html.escape(text)}</p>
          <p>全零 chirp：{q['all_zero_ADC_chirps']}；接近 ADC 极限：{q['ADC_near_limit_fraction']:.3%}；
          尾部不足一帧：{q['trailing_incomplete_frame_bytes']} bytes。</p>
          <p>共用接触会话：{html.escape(rec['contact_id'])}；完整包 {contact['complete_packets']} 个。
          配对层级：{html.escape(rec['pairing_level'])}。</p></article>''')
    contact_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{q['complete_packets']}</td>"
                           f"<td>{q['incomplete_candidates']}</td><td>{q['issue_count']}</td>"
                           f"<td>{q['packets_constant_all_axes']}</td></tr>" for k, q in contacts.items())
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R2 本地预处理检查</title>
    <style>body{font:15px/1.7 system-ui,sans-serif;margin:28px auto;max-width:1250px;color:#17324a;padding:0 20px}
    h1{font-size:26px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:16px}
    article{border:1px solid #cfdae4;border-radius:8px;padding:16px}h3{font-size:15px;overflow-wrap:anywhere}
    svg{width:100%;font:12px system-ui}.warn{color:#9d3900;background:#fff4e7;padding:8px}
    .note{color:#42576a}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #ccd6df;padding:6px 12px}
    p{margin:8px 0}</style><h1>R2 本地预处理检查</h1>
    <p>蓝线：录制开头最多 1 秒的帧内变化能量；绿区：按实测距离限定的搜索区；橙线：选取峰。
    图上每条曲线分别归一化，仅便于查看位置，不能据此比较不同录制的绝对强弱。</p>
    <p>这是数据检查报告，不是分类结果或无丢包证明。距离 FFT 补零网格约 2 cm，真实分辨能力约 4 cm。
    有提示的记录保留并复核，不依据类别或分类成绩删除。</p>
    <h2>接触式长记录（每个原始 DAT 只计一次）</h2><table>
    <tr><th>接触会话</th><th>完整包</th><th>不完整候选包</th><th>格式问题</th><th>三轴全常量包</th></tr>'''
    page += contact_rows + '</table><h2>各段雷达</h2><div class="grid">' + "\n".join(cards) + '</div></html>'
    dest = root / "qc_report.html"
    dest.write_text(page, encoding="utf-8")
    print(f"Report: {dest}; {len(paths)} radar recordings, {len(contacts)} distinct contact sessions")


if __name__ == "__main__":
    main()
