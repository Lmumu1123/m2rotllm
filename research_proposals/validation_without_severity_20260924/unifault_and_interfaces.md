# UniFault、多模型验证和特征转换：核验说明

核验日期：2026-09-24。本说明只读核对已有代码、结果和作者资料，没有新增训练，也没有修改旧模型。

## UniFault 是什么，为什么选择它

UniFault 是针对振动故障诊断的预训练特征模型，不是大语言模型。它把时域信号切成小片段，交给 Transformer，输出表示这段振动的数字向量。论文采用自监督对比预训练：同一信号经增强得到的两个版本应具有相近表示。论文报告超过 690 万预训练样本，来源包括 CWRU、PU、IMS、MFPT、XJTU-SY 等故障数据；这属于作者报告，并非我们复现了它的预训练过程。[论文方法与实验](https://arxiv.org/html/2504.01373v2)

名称有一处值得澄清：论文 v2 表 2 将小版本称为 **Lite**，作者 GitHub 的公开下载称为 **Tiny**。本地用的是 GitHub Tiny 权重；核验配置为 128 维、4 层、4 个注意力头，823376 个参数。论文 Lite 的对应配置为 128／4／4、约 823K。[作者仓库](https://github.com/emadeldeen24/UniFault)

选择它的理由是：有可下载并核验的真实预训练权重；专门处理接触振动；时域 Transformer 与 Bear／Rot 的频域卷积结构有明显区别；小模型方便完整验证。它用于增加教师结构的多样性，不是因为先在本地横评很多模型后只留下最高分者。已有实验前协议记录了教师、主输入和全部敏感性分支，但不应将协议解读为已比较了所有候选模型。

## 实际训练了什么

| 部分 | 本轮是否用本地数据训练 | 实际工作 |
|---|---|---|
| UniFault 预训练特征网络 | 否，冻结 | 加载公开权重，保持参数不变 |
| 接触输入归一化 | 拟合训练录制统计量 | 每折只用训练接触数据确定 min-max |
| UniFault 本地四分类头 | 是 | 冻结特征上拟合 StandardScaler＋四分类逻辑回归 |
| UniFault 目标雷达 encoder | 是，每折每方法每种子独立训练 | 学习将雷达特征变为 UniFault 可读取的表示 |
| UniFault 原生 16 输出头 | 未修改，也未用于报告原生四分类成绩 | 未核验到 16 类语义映射，不能编造映射 |

所以“UniFault 接触模型重训后达到 100%”不够准确。准确说法是：**冻结 UniFault 官方特征网络，用本地接触训练数据新建四分类头，留出接触窗口 58/58 正确，录制 8/8 正确。**

这不能直接推出 UniFault 本体优于 BearLLM／RotLLM。三者输入和本地适配路线不同；Rot 输入还有已报告的尺度歧义；58 个窗口只来自 8 次录制，每类两次录制是同一轴承。与此同时，三个简单接触 RMS 指标就达到 56/58，12 个幅度统计指标达到 57/58。当前本地任务还不能区分强大的故障预训练知识与容易区分类别的统计差异。UniFault 主分支对其他电机 bigNormal 的两个折分别为 0/12 和 12/12，也说明主四类 100% 不是广泛泛化证明。

本地核验：[接触结果](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/teachers/unifault/pooled_metrics.csv)、[简单统计对照](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/teachers/contact_sanity_controls/pooled_metrics.csv)、[预先固定的处理协议](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/teachers/protocol_before_results.json)、[来源与权重完整加载](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/teachers/unifault/provenance.json)。

## 三种容易混淆的验证

| 路线 | 雷达 encoder 是否重训 | 它主要验证什么 | 当前是否完成 |
|---|---|---|---|
| 固定某教师，重新训练雷达 encoder | 是 | 同一训练思路能否服务不同接触模型 | Bear、Rot、UniFault 均已做 |
| 冻结 Bear 目标雷达 encoder，加接触数据拟合的转换，接 Rot 头 | 否 | 现有雷达表示能否复用到另一套读取坐标 | 已做 Bear→Rot |
| 冻结 Bear 目标雷达 encoder，接 UniFault | 否 | 同上，但教师结构差异更大 | **没有这一实验结果** |

“同一训练框架”指雷达输入、学生主体网络、数据划分、步数与主要损失安排相近；不指三套训练出来的雷达权重相同。UniFault 还需要将学生最后的非负限制改成允许正负输出。因此，不能说“多模型实验中雷达 encoder 完全不用变”。

前一种实验支持训练方法能够接入三种教师；第二种才检验冻结表示的复用。它们都没有仅凭当前四分类成绩证明可迁移丰富故障知识。尤其不能把已完成的 Bear→Rot 扩写成 Bear→Rot／UniFault 均免重训。

## 转换模块是什么，它怎样处理雷达

这是一层带正则约束的线性回归（Ridge，固定 alpha=1），不是另一个雷达信号处理器。

训练转换时，对同一份**接触训练窗口**，分别运行 Bear 和 Rot，得到一对 128 维向量；拟合一个从 Bear 坐标到 Rot 坐标的转换。标准化和逆标准化最终可以合并成矩阵乘法与偏置，再以 ReLU 保持 Rot 特征的非负范围：

`转换后向量 = max(W × Bear坐标向量 + b, 0)`

转换训练不使用雷达，不读取测试录制，也不向回归器提供类别标签；但后面的新 Rot 四分类头使用过本地接触训练标签。不能把整条系统称为完全不使用本地标签。

雷达推理时：

`雷达信号 → 原雷达预处理 → 已冻结雷达 encoder → 上述转换 → Rot 分类头`

转换处理的是 **encoder 输出的 128 个数**，不会再次解码 IQ、改采样率、改时频图或运行 SFN。雷达 encoder 权重未变，但部署系统确实多了一层转换。

“直接 Rot 接触 8/8、Bear 接触转换为 Rot 是 7/8”的意思是：同样 8 条测试接触录制，直接用 Rot 自己的特征，新 Rot 头全部判断正确；先用 Bear 提特征，再近似转换成 Rot 坐标，只有 7 条正确。两条路径都使用接触数据，这个检查是为了证明转换器本身并不完美，**不是说雷达只答对 7 条**。

对应窗口成绩为直接 Rot 49/58，Bear→Rot 46/58。录制成绩通过同一条录制所有窗口的类别概率求均值得到，所以不能把 8/8 写成每秒窗口全对。

本地核验：[实际转换脚本](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/scripts/run_corrected_stitch.py)、[路径逐项成绩](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/results/summary/stitch.csv)。

## “准确率掩盖接口错误”是什么意思

Bear／Rot 的目标特征经过 ReLU，只包含非负值；UniFault 的特征有正有负。如果仍强制雷达输出非负数，例如教师想要 −2、学生只允许输出大于等于 0，学生再努力也不可能精确表示这个维度。

但分类头不要求每个数都一模一样。很多不同向量会被它判为同一类。加入类别损失后，学生可以找到一个全为非负、却依然让分类头输出正确类别的向量。因此“分类答对”与“表示接口正确、特征匹配好”并不等价。

已有 UniFault 实验：

| 雷达输出范围 | 仅接触均值匹配窗口准确率 | 完整方法窗口准确率 |
|---|---:|---:|
| 非负，无法表达教师负值 | 47.25% | 99.68% |
| 允许正负，满足教师接口 | 99.84% | 99.84% |

99.68% 不是测试标签泄露的证据；该训练仍只使用训练集合。它说明类别监督可以帮助完成当前分类，却不保证学到了要求复制的教师表示。只看准确率就会漏掉这个实现错误。

这项观察对框架有实际价值：学生输出的维度、符号范围和数值尺度必须与教师一致；训练与验收同时检查分类结果和特征误差。不能因分类接近满分就省略接口核验。

本地核验：[两种输出接口训练实现](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/scripts/train_unifault_radar.py)、[完整统计](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/results/summary/unifault_radar.csv)。表中窗口数据跨三个种子汇总，相同物理窗口重复出现，不代表增加了三个独立采集样本。
