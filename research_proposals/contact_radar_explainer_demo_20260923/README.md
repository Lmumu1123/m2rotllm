# 预处理解释与真实模型 demo

- [逐项解释](/home/huangyating/research_proposals/contact_radar_explainer_demo_20260923/预处理与对齐逐项解释.md)：IQ、频带输入、窗口、DCT/DC/BN、query/reference、bag、49.8% 和 99% 的口径、环境表格。
- `replay/模型输入输出演示.html`：自包含页面，下载后直接用浏览器打开，无需服务器、网络或基础模型。
- `replay/demo_data.json` / `sample_predictions.csv`：实际模型输出。
- `replay/*.npz`：每个样例的 DCN、雷达特征、hidden 和概率。
- `run_demo.py`：在当前服务器 m2vllm 中真正运行模型；不是训练脚本。

默认每份录制取首个有效窗口，与预测对错无关。当前 12 个例子保留正常/内圈/外圈/滚动体、未知底座故障和外部正常电机；错误 ROI 会明确标为不可发布诊断。

## 重新运行

输出目录必须是新的或空的，不覆盖已生成演示：

```bash
/home/huangyating/miniconda3/envs/m2vllm/bin/python \
  /home/huangyating/research_proposals/contact_radar_explainer_demo_20260923/run_demo.py \
  --output /home/huangyating/research_proposals/contact_radar_explainer_demo_20260923/replay_repeat
```

只跑指定录制可加：

```text
--bag 20260920-193718-roll-115200
```

依赖当前机器的 BearLLM FCN 权重、适配头、雷达模型、原有 NPZ 和固定九个公开健康参考，路径在脚本中明确。若只需向导师展示，直接打开已经生成的 HTML 即可；压缩包不包括上述大型权重。

## 推理边界

雷达输入从已归一化和去帧均值的距门 IQ 开始，并非完整原始 ADC/bin 解码。接触输入是保存的一秒 XYZ。两个首窗口不保证物理上同时；不能拿曲线直接判断真实时间偏移。

推理只读取信号、固定 scaler/模型和健康参考，标签在之后用于结果对照。没有拟合测试 scaler，没有重新训练，没有实际调用 Qwen。页面显示四类概率和分类头预测，不把固定类名展示冒充语言模型生成。

主录制采用两个对应的留出模型，外部录制采用保守 all_known 模型；12 例不是一个新测试集。逐例来源、数值误差、输入哈希和推理核验见 `replay/verification.json`。
