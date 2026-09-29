"""Derive a concise, human-fillable plan from the audited full manifest."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'collection_manifest_main80.csv'
OUTPUT = ROOT / 'collection_operator_main80.csv'

PLANNED = {
    'plan_id': '计划编号',
    'session_plan': '计划会话',
    'planned_order': '计划顺序',
    'fault_type_plan': '故障类型',
    'rpm_setpoint_plan': '计划转速rpm',
    'environment_plan': '计划环境',
    'distance_cm_plan': '计划距离cm',
    'split_fixed_speed': '固定转速协议用途',
    'target_contact_complete_packets': '目标完整接触包数',
}
ACTUAL = ['实际录制ID', '实际日期时间与时区', '轴承实体ID', '安装与姿态ID',
          '实际cfg版本ID', '实际距离cm', '实际转速证据_反馈或setpoint_only',
          '转速日志或实测范围', '接触原文件路径', '雷达原文件路径',
          '日志与照片目录', '失败或重采关联ID', '安装负载环境改动说明']

def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        source = list(csv.DictReader(handle))
    assert len(source) == 80 and len({r['plan_id'] for r in source}) == 80
    with OUTPUT.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLANNED.values()) + ACTUAL)
        writer.writeheader()
        for row in source:
            writer.writerow({**{label: row[key] for key, label in PLANNED.items()},
                             **dict.fromkeys(ACTUAL, '')})
    print(f'Wrote {len(source)} planned rows to {OUTPUT}; actual fields remain blank.')

if __name__ == '__main__':
    main()
