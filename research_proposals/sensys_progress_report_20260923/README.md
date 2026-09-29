# 导师汇报材料

日期：2026-09-23。实验结果截至 2026-09-22。本次只汇总、核对与制图，没有重新训练模型或修改历史结果。

## 建议阅读顺序

1. `导师汇报_10分钟提纲.md`：8 页讲述安排，可用于组织 PPT；不是实际 PPT 文件。
2. `导师汇报_接触式知识迁移到毫米波诊断与SenSys计划.docx`：适合发给导师并批注的完整 Word 文档。
3. 同名 `.md`：可继续修改的主稿，包含算法公式、结果表、引用和证据入口。
4. 同名 `.html`：浏览器阅读/打印版，图片已嵌入，正文无需联网；外部文献及服务器证据链接需相应访问条件。

Word 的字体回退和分页取决于阅读设备；HTML 可使用浏览器“打印→另存为 PDF”。本次不把未进行版面渲染的 Word 称为已验收的排版 PDF。

## 文件与范围

- `figures/`：彩色流程和实测结果图，PNG/PDF/SVG 三种格式；图中的结果来自归档 CSV。
- `evidence/`：主要数字的 CSV/JSON 快照、来源路径、SHA256 和本次核验记录。不是全部实验原始数据或模型。
- `scripts/build_report.py`：制图、文档转换与关键计数交叉核验脚本；不运行训练。
- `.render_dependencies/`：仅用于本文档生成的隔离依赖，不改动 m2vllm 的训练依赖；不纳入交付压缩包。

主稿明确区分三代模型、训练与留出、窗口与录制、旧能力回归与新域泛化。跨电机接触特征泛化、正确 ROI 下的四类雷达验收、严格时间配准及迁移优于 baseline 均标为未完成。

## 复现生成

在当前服务器目录内：

```bash
/home/huangyating/miniconda3/envs/m2vllm/bin/python /home/huangyating/research_proposals/sensys_progress_report_20260923/scripts/build_report.py
```

若从压缩包重新构建，先在新目录的 `.render_dependencies` 中安装 `markdown`、`python-docx`（连同 lxml 依赖），并准备 matplotlib。脚本中的原始证据路径指向当前服务器，异地使用需修改来源路径；现成 Word/HTML 阅读不依赖这些软件或模型。

压缩包包含文稿、图和小型证据快照，不包含传感器原始数据、权重或大型缓存。
