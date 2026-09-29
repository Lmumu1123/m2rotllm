#!/usr/bin/env python3
"""按已确认的文件命名生成本地处理清单，不解析传感器数据。

Mac 使用示例：
  python build_manifest.py --data-root "/Users/anthea/Desktop/new_data" \
      --output manifest.csv

已知采集协议：同故障、同转速的 DAT 持续覆盖该工况下全部距离和重复的
雷达录制。因此唯一 DAT 可以由多个雷达共享；这不表示逐窗口精准同步。
文件名时间只作临时重复编号，不能证明独立实验或精确时间对应。
多个 DAT 候选时不按最近时间猜测；请查看自动生成的 review.csv。
"""
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import argparse
import csv
import json
import re


FIELDS = ["recording_id", "label", "rpm", "distance_cm", "repeat", "radar_path",
          "radar_parts", "contact_path", "paired_confirmed", "session_id", "bearing_id",
          "environment", "contact_fs_hz", "contact_unit", "cfg_path", "radar_format",
          "iq_order", "aux_paths", "notes", "contact_id", "pairing_level"]
REVIEW_FIELDS = ["issue", "severity", "modality", "path", "label", "rpm", "distance_cm",
                 "recording_id", "candidate_contact_id", "details"]
LABELS = ["normal", "inBroken", "outBroken", "roll"]
ALIASES = {"normal": "normal", "in": "inBroken", "inbroken": "inBroken",
           "inner": "inBroken", "out": "outBroken", "outbroken": "outBroken",
           "outer": "outBroken", "roll": "roll", "ball": "roll"}
RADAR_RE = re.compile(r"(?P<stamp>\d{8}-\d{6})-(?P<label>[A-Za-z]+)-"
                      r"(?P<rpm>\d+)r-(?P<cm>\d+)cm(?:-rep(?P<rep>\d+))?", re.I)
CONTACT_RE = re.compile(r"(?P<stamp>\d{4}_\d{1,2}_\d{1,2}_\d{1,2}-\d{2}-\d{2})-"
                        r"(?P<label>[A-Za-z]+)-(?P<rpm>\d+)r", re.I)


def parse_file(path, root):
    modality = "radar" if path.suffix.lower() == ".bin" else "contact"
    pattern = RADAR_RE if modality == "radar" else CONTACT_RE
    match = pattern.fullmatch(path.stem)
    if match is None:
        return None, dict(issue="unrecognized_filename", severity="review", modality=modality,
                          path=path.relative_to(root).as_posix(), details=(
                              "不自动忽略或猜测此文件。若为 DCA 分卷，须先确认同一次采集及分卷顺序，"
                              "再通过 radar_parts 显式登记；不能把分卷当独立重复。"))
    values = match.groupdict()
    try:
        label = ALIASES[values["label"].lower()]
        stamp = datetime.strptime(values["stamp"], "%Y%m%d-%H%M%S" if modality == "radar"
                                  else "%Y_%m_%d_%H-%M-%S")
        rpm = int(values["rpm"])
        cm = int(values["cm"]) if modality == "radar" else None
        rep = int(values["rep"]) if values.get("rep") else None
        if rpm <= 0 or (cm is not None and cm <= 0) or (rep is not None and rep <= 0):
            raise ValueError("转速、距离、重复编号必须为正")
    except (KeyError, ValueError) as exc:
        return None, dict(issue="invalid_filename_metadata", severity="review", modality=modality,
                          path=path.relative_to(root).as_posix(), details=str(exc))
    return dict(path=path.relative_to(root).as_posix(), modality=modality, label=label,
                rpm=rpm, cm=cm, explicit_repeat=rep, stamp=stamp,
                size_bytes=path.stat().st_size), None


def generate(root):
    radar, contact, review = [], [], []
    counts = Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".bin", ".dat"}:
            continue
        counts[path.suffix.lower()] += 1
        item, issue = parse_file(path, root)
        if issue:
            review.append(issue)
        else:
            (radar if item["modality"] == "radar" else contact).append(item)
    contacts = defaultdict(list)
    groups = defaultdict(list)
    for item in contact:
        contacts[(item["label"], item["rpm"])].append(item)
    for key, items in contacts.items():
        items.sort(key=lambda item: (item["stamp"], item["path"]))
        for i, item in enumerate(items, 1):
            suffix = "" if len(items) == 1 else f"_candidate{i}"
            item["contact_id"] = f"{key[0]}_{key[1]}r_session{suffix}"
        if len(items) > 1:
            for item in items:
                review.append(dict(issue="multiple_contact_candidates", severity="review", modality="contact",
                                   path=item["path"], label=key[0], rpm=key[1],
                                   candidate_contact_id=item["contact_id"], details=(
                                       "同故障/转速有多个 DAT；未根据时间邻近猜测对应。"
                                       "请人工确认每个雷达被哪个 DAT 覆盖，再填写 contact_path/contact_id/paired_confirmed。")))
    for item in radar:
        groups[(item["label"], item["rpm"], item["cm"])].append(item)
    rows = []
    for key in sorted(groups, key=lambda k: (LABELS.index(k[0]), k[1], k[2])):
        items = sorted(groups[key], key=lambda item: (item["stamp"], item["path"]))
        repeats = [item["explicit_repeat"] for item in items if item["explicit_repeat"] is not None]
        duplicate_repeats = {x for x, n in Counter(repeats).items() if n > 1}
        used = set(repeats)
        ordinal = 1
        for item in items:
            if item["explicit_repeat"] in duplicate_repeats:
                review.append(dict(issue="duplicate_explicit_repeat", severity="review", modality="radar",
                                   path=item["path"], label=key[0], rpm=key[1], distance_cm=key[2],
                                   details="同工况的显式 rep 编号重复；这些文件未写入处理清单，请核验重命名或分卷关系。"))
                continue
            rep = item["explicit_repeat"]
            if rep is None:
                while ordinal in used:
                    ordinal += 1
                rep = ordinal
                used.add(rep)
                repeat_note = "repeat由文件名时间排序临时编号，不证明独立重新采集或跨距离同一轮次"
            else:
                repeat_note = "repeat来自文件名显式rep编号，独立采集关系仍须由实验记录确认"
            candidates = contacts.get(key[:2], [])
            ref = candidates[0] if len(candidates) == 1 else None
            rid = f"{key[0]}_{key[1]}r_{key[2]}cm_rep{rep}"
            notes = [repeat_note, "接触式采样率4000Hz已由用户确认", "4k包间缺口未知",
                     "文件名为接收开始时间，不能据此精准配窗", "radar_format须由实际采集方式确认后在命令行指定"]
            if ref:
                notes.append("用户确认该DAT连续覆盖同故障同转速的全部距离和雷达重复；paired_confirmed仅表示此采集会话覆盖，不表示逐窗对齐")
            else:
                notes.append("接触DAT对应未解决，处理前人工核验；不能据时间邻近推断")
                review.append(dict(issue="missing_contact" if not candidates else "unresolved_contact_mapping",
                                   severity="review", modality="radar", path=item["path"], label=key[0],
                                   rpm=key[1], distance_cm=key[2], recording_id=rid,
                                   details="未找到唯一同工况DAT，contact_path/contact_id/paired_confirmed留空。"))
            rows.append(dict(recording_id=rid, label=key[0], rpm=key[1], distance_cm=key[2], repeat=rep,
                             radar_path=item["path"], contact_path=ref["path"] if ref else "",
                             contact_id=ref["contact_id"] if ref else "", contact_fs_hz=4000,
                             paired_confirmed="yes" if ref else "", iq_order="iiqq_neg",
                             pairing_level="continuous_session", notes="；".join(notes)))
        if len(items) != 4:
            review.append(dict(issue="repeat_count_differs_from_four", severity="info", modality="radar",
                               label=key[0], rpm=key[1], distance_cm=key[2],
                               details=f"该工况识别到{len(items)}份bin，计划为4份；允许先处理小批，需核对全量采集。"))
        stamps = Counter(item["stamp"] for item in items)
        if any(n > 1 for n in stamps.values()):
            review.append(dict(issue="duplicate_filename_timestamp", severity="review", modality="radar",
                               label=key[0], rpm=key[1], distance_cm=key[2],
                               details="同工况存在相同文件名时间；核验是否重复复制或同一采集分卷，不能视为独立重复。"))
    for key, items in contacts.items():
        if not any(group[:2] == key for group in groups):
            for item in items:
                review.append(dict(issue="contact_without_radar_condition", severity="info", modality="contact",
                                   path=item["path"], label=key[0], rpm=key[1],
                                   candidate_contact_id=item["contact_id"], details="DAT已识别，但没有同故障/转速雷达文件。"))
    expected = [(label, rpm, cm) for label in LABELS for rpm in [1000, 2000, 3000] for cm in [20, 40, 80]]
    summary = dict(schema="r2-manifest-builder-20260928-v1", data_root=str(root),
                   discovered_bin_files=counts[".bin"], discovered_dat_files=counts[".dat"],
                   recognized_radar_files=len(radar), recognized_contact_files=len(contact),
                   manifest_rows=len(rows), mapped_radar_rows=sum(bool(row["contact_path"]) for row in rows),
                   unique_mapped_contact_ids=len({row["contact_id"] for row in rows if row["contact_id"]}),
                   planned_radar_recordings=144, expected_contact_sessions_if_one_per_condition=12,
                   all_expected_radar_conditions_have_four=(all(len(groups.get(key, [])) == 4 for key in expected)
                                                           and len(radar) == 144 and len(rows) == 144),
                   missing_radar_conditions=[dict(label=k[0], rpm=k[1], distance_cm=k[2]) for k in expected if k not in groups],
                   contact_candidates=[dict(contact_id=item["contact_id"], path=item["path"], label=item["label"],
                                            rpm=item["rpm"], size_bytes=item["size_bytes"])
                                       for key in sorted(contacts) for item in contacts[key]],
                   review_issue_counts=dict(Counter(row["issue"] for row in review)),
                   review_required=any(row["severity"] == "review" for row in review),
                   notes=["文件名匹配是清单辅助，不证明字节排列、DCA完整性或精确时序。",
                          "共享contact_id的雷达不得被当作拥有独立接触教师数据；划分和报告必须考虑共享来源。",
                          "所有split尚未指定；repeat时间排序并不自动建立训练/验证/测试独立性。",
                          "可先处理小批；144/12为计划数量检查，不是运行前强制门槛。"])
    return rows, review, summary


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="新生成的manifest.csv；不覆盖已有文件")
    parser.add_argument("--review-output", type=Path, help="默认同目录 manifest_review.csv")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    review_path = (args.review_output.expanduser().resolve() if args.review_output else
                   output.with_name(output.stem + "_review.csv"))
    summary_path = output.with_name(output.stem + "_summary.json")
    if not root.is_dir():
        parser.error(f"数据目录不存在：{root}")
    targets = [output, review_path, summary_path]
    if len(set(targets)) != 3 or any(path.exists() for path in targets):
        parser.error("输出路径必须彼此不同且均不存在；请使用新文件名以保留此前人工编辑。")
    rows, review, summary = generate(root)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows, FIELDS)
    write_csv(review_path, review, REVIEW_FIELDS)
    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"清单：{output}（{len(rows)}条雷达；映射{summary['unique_mapped_contact_ids']}个接触会话）")
    print(f"核对表：{review_path}；数量统计：{summary_path}")
    if summary["review_required"]:
        print("存在需核对的文件或映射，请先查看核对表；未自动猜测不明确关联。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
