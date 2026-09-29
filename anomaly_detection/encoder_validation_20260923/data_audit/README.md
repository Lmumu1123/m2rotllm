# 原始数据接收与统一距离门重处理

用户已确认原始文件位于 Mac：

```text
/Users/anthea/Desktop/C2RLLM/data接触式和非接触式同时采集
```

当前是 Linux 服务器环境，不能直接读取该 Mac 路径。原始文件应同步到：

```text
/home/huangyating/anomaly_detection/raw_four_class
```

本目录没有自动传输程序或后台监控。先手动同步完整文件，然后运行单次检查：

```bash
/home/huangyating/miniconda3/envs/m2vllm/bin/python /home/huangyating/anomaly_detection/encoder_validation_20260923/data_audit/receive_and_reprocess.py
```

默认只检查 12 个 bin 的文件名、重复、归档大小、完整帧与帧尾、文件稳定性，以及归档配置／接触导出的存在性。DAT 是可选输入。`--verify-sha256` 可在检查阶段增加完整内容哈希。

检查结果保存为 `upload_readiness.json`。返回码 2 表示当前不能重处理，报告列出原因；返回码 0 表示检查通过。默认检查通过不表示已经完成原始 SHA256 核验。

实际重处理需要显式加 `--run`：

```bash
/home/huangyating/miniconda3/envs/m2vllm/bin/python /home/huangyating/anomaly_detection/encoder_validation_20260923/data_audit/receive_and_reprocess.py --run
```

该操作调用原有 `four_class_chain_20260922/scripts/reprocess_geometry.py`，导出前强制核验全部原始 bin SHA256。候选距离范围统一为 0.45±0.16 米，随后按照同一无标签能量规则选取距离单元；不改配置、不访问雷达、不训练模型。

默认写入新的 `encoder_validation_20260923/geometry_corrected_r0_045m`。已有非空输出会被拒绝；再次运行需要通过 `--output` 指定另一个新目录。

已在当前空接收目录实际执行默认命令：找到 0/12 个 bin，退出码为 2，生成缺失报告，未生成修复数据目录。完整处理尚未执行。

详细数据边界、旧外圈 ROI 的限制和同类错配可行性，见 `数据可用性与ROI检查.md`。
