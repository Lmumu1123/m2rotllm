# 20 cm 雷达：扩大距离备份的本地补导方案

本轮已跑完现有数据的实验，这个补导尚未执行，原因是原始 BIN 在 Mac。服务器现有宽 ROI 不能恢复其外部的复数相位。

审计发现 20 cm 录制多数全距离动态峰约为 0.321–0.341 m，现有宽 ROI 最大离散 bin 约 0.301 m。现在不能仅凭标记距离判定这些峰是否就是电机，需要确认实际测距基准；扩大备份用于复核，并不保证分类提高。

`manifest_20cm_48.csv` 从上传的 `recording.json/metadata` 派生，共 48 条，保留原相对文件名及人工确认的 `20m→20cm` 元数据。它不是原始 Mac manifest 的校验副本，不包含本地未上传的其他手工记录。

将本文件夹放到 Mac，把 CSV 复制到原来可运行的工具包目录。在原工具虚拟环境下运行：

```bash
python preprocess_local.py process \
  --data-root "/Users/anthea/Desktop/C2RLLM/data大量测试" \
  --manifest manifest_20cm_48.csv \
  --cfg R2_reference.cfg \
  --radar-format headerless_reordered_adc \
  --roi-half-width-m 0.24 \
  --output "/Users/anthea/Desktop/C2RLLM/R2_20cm_wide_v2"

python make_qc_report.py \
  --output "/Users/anthea/Desktop/C2RLLM/R2_20cm_wide_v2"

python preprocess_local.py verify \
  --output "/Users/anthea/Desktop/C2RLLM/R2_20cm_wide_v2"
```

这会将备份范围扩至约 0.02–0.44 m；近端硬件泄漏可能也在范围里，不能把其中任意最强峰自动认定为电机。全部 48 条采用同一个几何范围，不看类别或分类成绩来选择。

补传重点是 `wide_cache_local/` 的 48 条文件，以及新的 `qc_report.html/qc_index.csv/summary.json` 和 `upload/radar/*/recording.json`，存到服务器新目录，保留目前原导出结果。若接触式源文件没有变化，不需要再次传同一 DAT。

新结果用于开发性选区复核；本轮原始成绩不会被替换成“第一次盲测”的成绩。若确认实测几何后应使用其他范围，再先固定统一范围并重新导出。
