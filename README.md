# openpi 深度学习参考库

> 面向 **VLA（Vision-Language-Action）小白** 的 Physical Intelligence (π) / openpi 系统化深度参考库。
>
> 把官方源码 + 全套中文逐主题深化文档放在同一个仓库里，开 IDE 即可"边读文档边跳转源码"。

---

## 仓库结构

```
openpi-deepdive/
├── README.md                       # 本文件（导航）
├── docs/                           # 10+ 篇逐主题深化中文文档
│   ├── 00-overview.md              # 全景概览（公司、VLA、版本时间线、checkpoint 矩阵）
│   ├── 01-architecture-and-math.md # 架构数学：flow matching 公式推导 + 双塔 MoE attention
│   ├── 02-code-framework-map.md    # 代码框架地图：每个 .py 文件作用 + 关键函数行号
│   ├── 03-tech-stack.md            # 技术栈：每个依赖为什么选 + 版本矩阵
│   ├── 04-training-pipeline.md     # 训练 pipeline：数据加载 → 归一化 → FSDP → checkpoint
│   ├── 05-inference-and-serving.md # 推理 pipeline：Policy 封装 + WebSocket 远程部署
│   ├── 06-data-transforms.md       # 数据 transforms 三层管线 + ALOHA/DROID/LIBERO policy
│   ├── 07-key-techniques.md        # LoRA / AdaRMS / KV cache / gemma_fast 源码精读
│   ├── 08-version-source-diff.md   # π₀ vs π₀-FAST vs π₀.₅ vs π₀.₆ 源码 diff 对照
│   ├── 09-tokenizers.md            # PaliGemma SentencePiece + FAST (DCT+BPE) 完整算法
│   └── 10-finetune-cookbook.md     # 从零 fine-tune 你自己机器人的端到端食谱
├── references/
│   └── openpi/                     # git submodule：openpi 官方源码（用于 IDE 跳转）
├── examples/                       # （后续）最小可运行示例：训练/推理/远程客户端
└── assets/                         # 图示与示意图
```

`references/openpi/` 是 **git submodule** 指向 `https://github.com/Physical-Intelligence/openpi`。
克隆本仓库时记得用 `--recurse-submodules`：

```bash
git clone --recurse-submodules https://github.com/x111-kk/openpi.git
# 或者已经克隆完：
git submodule update --init --recursive
```

---

## 文档阅读顺序建议

| 你是谁                          | 建议阅读路径                                                |
| ------------------------------- | ----------------------------------------------------------- |
| **VLA 小白**（先建立宏观认知）  | `00` → `01` → `09` → `03`                                   |
| **想理解 π 是怎么训出来的**     | `00` → `01` → `04` → `06` → `07` → `10`                     |
| **想做远程推理 / 部署到机器人** | `00` → `05` → `06`                                          |
| **想 fork π₀ 改架构**           | `01` → `02` → `07` → `08` → `04`                            |
| **想 fine-tune 自己的机器人**   | `00` → `06` → `10` → `04`                                   |
| **想直接读源码**                | `02`（代码地图） + 在 `references/openpi/` 里跳转           |

每篇文档都给出了**确切的源码行号锚点**（形如 `src/openpi/models/pi0.py:189`），可直接在你的 IDE 里 Ctrl+Click 跳转。

---

## 核心结论（一句话总结）

- **π = 双塔 MoE Transformer + Flow Matching**：左塔 PaliGemma 3B VLM 处理 (图像 + 语言)，右塔 300M Action expert 处理 (state + 加噪 action chunk + timestep)，共享 attention，前向 10 步去噪生成 50 步动作块。
- **π₀.₅ 加 AdaRMS 时间条件 + Knowledge Insulation**，让 VLM 在 fine-tune 时不被 action expert 梯度污染。
- **π₀-FAST 用 DCT+BPE 把动作离散化成 token**，可作为纯自回归 LLM 训练（不需要 flow matching），代价是推理慢。
- **openpi 用 JAX/Flax NNX 写**（也提供 PyTorch 等价实现），Orbax 存 checkpoint，LeRobot 提供数据格式，WebSocket + msgpack 做远程推理。

---

## 学习参考

- 官方主页：<https://www.physicalintelligence.company/>
- 官方仓库：<https://github.com/Physical-Intelligence/openpi>
- π₀ 论文：<https://www.physicalintelligence.company/blog/pi0>
- π₀-FAST 论文：<https://www.physicalintelligence.company/research/fast>
- π₀.₅ 博客：<https://www.physicalintelligence.company/blog/pi05>
- Knowledge Insulation：<https://www.physicalintelligence.company/research/knowledge_insulation>
- Real-Time Chunking：<https://www.physicalintelligence.company/research/real_time_chunking>

---

> 本仓库的所有中文文档由 Devin 基于 openpi 源码逐文件阅读 + 官方博客原文交叉验证整理。源码归属于 Physical Intelligence，遵循其 Apache-2.0 协议。
