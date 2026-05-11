# Physical Intelligence (π) / openpi 深度解析

> 面向 VLA 小白的"从结构到应用"系统化讲解
> 资料来源：[openpi 仓库](https://github.com/Physical-Intelligence/openpi)、[pi.website](https://www.pi.website) 全部官方博客 / 论文（截至 2025-11）
> 作者整理：Devin（基于源码逐文件阅读 + 官方博客原文交叉验证）

---

## 0. 一句话总览

**Physical Intelligence（缩写 π，发音 "pi"）** 是 2024 年由 Sergey Levine、Chelsea Finn、Karol Hausman 等人创立的具身智能公司，目标是做"机器人界的 GPT"——**机器人基础模型（Robot Foundation Model）**。

**openpi** 是 π 公司开源的官方代码与模型权重仓库，目前包含 3 类共 6+ 个模型：

| 模型族 | 类型 | 一句话描述 |
|---|---|---|
| π₀ | flow-matching VLA | 第一代通用机器人策略，连续动作输出 |
| π₀-FAST | autoregressive VLA | 离散 token 化动作，5× 训练加速 |
| π₀.₅ | flow-matching VLA + KI | 升级版，跨家庭/跨场景开放世界泛化 |
| π₀.₆ / π*₀.₆ | π₀.₅ + Recap RL | 加入真实世界 RL，吞吐量翻倍（**未开源**，闭源产品线） |
| Hi Robot | 分层 VLA 系统 | 高层 VLM "System 2" + π₀ "System 1" |

**openpi 仓库目前开源**：π₀、π₀-FAST、π₀.₅ 的 base checkpoints + 多个机器人平台的 fine-tuned 权重 + JAX/PyTorch 完整训练/推理代码。

---

## 1. 公司与项目背景（小白快速入门）

### 1.1 Physical Intelligence 公司是谁？
- 总部：旧金山
- 创立：2024 年初
- 核心愿景：**Artificial Physical Intelligence（人工物理智能）**——让机器人像 ChatGPT 接受自然语言指令那样，做任何用户要求的物理任务
- 团队核心：Sergey Levine（UC Berkeley，RL 大牛）、Chelsea Finn（Stanford，元学习 / 模仿学习权威）、Karol Hausman（前 Google Brain，RT-1/RT-2 团队）、Brian Ichter（SayCan 作者）、Karl Pertsch、Danny Driess、Quan Vuong、Lucy Shi 等

### 1.2 什么是 VLA？（这是理解 openpi 的关键概念）

**VLA = Vision-Language-Action 模型**，一句话定义：

> 一个神经网络同时接收**图像（V）**、**语言指令（L）**，输出**机器人动作（A）**。

类比理解：
- **VLM**（视觉语言模型，如 GPT-4V、Gemini）：图像 + 语言 → 文本
- **VLA**：图像 + 语言 → 电机指令（关节角度、夹爪开合等）

VLA 的核心难点：
1. **VLM 只会输出离散 token**（"the", "cat"），但机器人要输出**高频连续控制量**（50Hz 的浮点关节角度）
2. **数据稀缺**：互联网上有海量图文，但机器人数据要"真人遥操作真机器人"才能获得，量级小 1000 倍
3. **延迟敏感**：VLM 推理几百毫秒没问题，机器人晚 100ms 可能就把咖啡洒到你腿上

π 公司的所有工作都是围绕"如何把 VLM 改造成能控制真实机器人的 VLA"展开的。

---

## 2. 版本迭代时间线（一图看懂 π 全家桶）

```
2024-10  π₀         首代 VLA（flow matching）             [开源 base]
2025-01  π₀-FAST    FAST 离散动作 tokenizer + AR VLA      [开源 base]
2025-02  Hi Robot   分层 VLA：高层 VLM + 低层 π₀          [开源论文]
2025-04  π₀.₅       跨家庭泛化、co-training 多模态数据    [开源 base, Sept 2025]
2025-05  KI         Knowledge Insulation 训练新方法       [并入 π₀.₅]
2025-06  RTC        Real-Time Chunking 实时推理算法       [推理时算法]
2025-09  PyTorch    openpi 添加 PyTorch 训练/推理         [代码]
2025-11  π₀.₆       Gemma3 4B backbone，更强 base         [闭源]
2025-11  π*₀.₆      π₀.₆ + Recap RL，企业级可靠性         [闭源]
```

**开源 vs 闭源**：openpi 仓库目前只放了 π₀、π₀-FAST、π₀.₅ 三个的 base 权重，**π₀.₆ / π*₀.₆ 暂未开源**（它们是 π 公司商业落地的核心，比如那段"机器人开了一天咖啡店"的演示视频用的就是 π*₀.₆）。

---

## 3. 核心架构（必读，所有版本的共同底座）

### 3.1 π₀ 的"VLM + Action Expert"双塔结构

```
      ┌────────────────────────────────────────────────────────────┐
      │              π₀ Network (Total ≈ 3.3B params)              │
      │                                                            │
      │   ┌─────────────────────┐      ┌──────────────────────┐    │
      │   │  PaliGemma VLM 主干  │      │  Action Expert       │    │
      │   │  (≈3B params)        │ ◄──► │  (gemma_300m, 300M)  │    │
      │   │                     │      │                      │    │
      │   │  - SigLIP ViT       │      │  - 同样 18 层 Transformer│    │
      │   │    (So400m/14)      │      │  - 但 width=1024     │    │
      │   │  - Gemma 2B LLM     │      │  - 处理动作 token     │    │
      │   │    (width=2048)     │      │  - 输出 flow vector  │    │
      │   └─────────────────────┘      └──────────────────────┘    │
      │                                                            │
      │  Prefix: 图像 patch tokens + 语言 prompt tokens             │
      │  Suffix: 状态 token + 噪声动作 tokens（flow matching 用）    │
      │                                                            │
      │  统一通过 mixture-of-experts 风格的 attention 在两塔间共享    │
      └────────────────────────────────────────────────────────────┘
```

#### 3.1.1 输入
- **多视角 RGB 图像**：典型 3 路：`base_0_rgb`（俯视/外部）、`left_wrist_0_rgb`（左手腕相机）、`right_wrist_0_rgb`（右手腕相机），分辨率 224×224（π₀.₆ 升到 448×448）
- **本体感觉状态 state**：机器人当前关节角度、夹爪位置等，维度统一 padding 到 32
- **语言 prompt**：自然语言指令，比如 "pick up the fork"

#### 3.1.2 输出
- **Action chunk**：一次预测一个**动作块**（不是单步动作），典型 chunk size = 50（对应 1 秒、50Hz 控制）
- 每个动作是 32 维向量（左右臂关节 + 夹爪 + 可选移动底盘 vx/vy）

#### 3.1.3 关键创新：双 Transformer 塔
- 一个 transformer block 内有 **两套权重**：VLM expert 和 Action expert
- Image / language token 走 VLM expert，action / state token 走 action expert
- 它们通过 **共享 attention** 互相"对话"
- 物理意义：VLM 不被电机控制信号污染（保留它的 web-scale 知识），动作专家可以专心学连续控制

> openpi 源码定位：<br>
> `src/openpi/models/pi0.py`（66-100 行）`class Pi0` 构造函数<br>
> `src/openpi/models/gemma.py` 的 `Module(configs=[paligemma_config, action_expert_config])` 实现双塔混合

### 3.2 Flow Matching：连续动作的核心训练目标

#### 小白版解释
Flow matching 是 diffusion model 的兄弟。直觉上：
- 取一段真实动作 `a` 和高斯噪声 `ε`
- 在它们之间任取一个时刻 `t∈[0,1]`，线性插值得到 `x_t = t·ε + (1-t)·a`
- **训练模型**：给定 `x_t` 和当前时刻 `t`，预测"从噪声指向真实动作的方向" `u_t = ε - a`
- **推理时**：从纯噪声出发，沿模型预测的"流场"积分 N 步（默认 10 步），就得到干净的动作 chunk

#### 与 diffusion 的区别
- Diffusion 学的是 score function（带条件高斯）
- Flow matching 学的是**矢量场**，目标函数更简单（直接 MSE），无 SNR 加权，训练更稳

> 源码：`src/openpi/models/pi0.py` 188-243 行 `compute_loss` 与 `sample_actions`

### 3.3 关键超参对照表

| 参数 | π₀ | π₀-FAST | π₀.₅ | π₀.₆ |
|---|---|---|---|---|
| VLM 主干 | PaliGemma 3B (Gemma 2B + SigLIP) | 同左 | 同左 | **Gemma3 4B** |
| 视觉编码器 | SigLIP So400m/14 | 同 | 同 | 同（448×448） |
| Action expert | gemma_300m (18L, w=1024) | — | gemma_300m | **~860M, 同层数** |
| Action dim | 32 | 32 | 32 | 32 |
| Action horizon (chunk) | 50 | 32 | 50 (DROID:15, Libero:10) | 50 |
| Max token len | 48 | 250 | 200 (state→discrete) | — |
| 动作生成方式 | flow matching | autoregressive FAST tokens | flow matching | flow matching + FAST aux |
| Timestep 注入 | concat MLP | — | **adaRMSNorm** | adaRMSNorm |
| 状态输入 | 连续 token (suffix) | 离散 token (prefix) | **离散 prompt token** | 离散 prompt token |
| 推理速度 | 快（10 步去噪） | 慢（AR 解码） | 快 | 快 |
| 语言跟随能力 | 中 | 强（VLM 不动） | 强（KI） | 强 |

---

## 4. 各版本逐个细讲

### 4.1 π₀（2024-10）—— 第一代通用机器人策略

**论文**：[arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
**博客**：[pi.website/blog/pi0](https://www.physicalintelligence.company/blog/pi0)

#### 关键贡献
1. 第一个真正能"通用"的 VLA：单一权重直接控制 **7 种机器人平台 + 68 个任务**
2. 把 PaliGemma 3B 改造成 VLA，通过 flow matching 输出 50Hz 连续动作
3. 预训练数据：**自有 8 种机器人 10000+ 小时数据 + 开源 OXE (Open X-Embodiment) 数据集**
4. 演示任务：折叠衣服（从烘干机→桌子→叠好成堆）、收拾餐桌、装杂货、打开爆米花袋……

#### 训练范式
```
Step 1: 大规模混合预训练（pre-training）
  ├── Internet-scale VLM (PaliGemma 已预训练好)
  ├── 8 种机器人的演示数据（自家收集）
  └── 公开机器人数据集 OXE

Step 2: 任务后训练（post-training）
  └── 在某个具体任务上 fine-tune（如 fold-laundry）
      使用更小但更高质量的演示数据集
```

#### checkpoint
- `gs://openpi-assets/checkpoints/pi0_base` —— 通用预训练 base，供你 fine-tune

---

### 4.2 π₀-FAST（2025-01）—— 用 FAST tokenizer 把动作变成 token

**论文/博客**：[pi.website/research/fast](https://www.physicalintelligence.company/research/fast)
**核心想法**：把"连续动作"压缩成"离散 token"，让 VLA 变成纯 autoregressive transformer，复用 LLM 的所有训练/推理工具链。

#### FAST tokenizer 工作流程

```
[a_0, a_1, ..., a_49]   (50 步 × 32 维 = 1600 浮点数)
       │
       ▼  Discrete Cosine Transform (DCT)
[频域系数矩阵]   (信号压缩：能量集中在低频)
       │
       ▼  Quantize（量化）
[整数稀疏矩阵]
       │
       ▼  按"低频优先"展平
[1D 整数序列，大量 0]
       │
       ▼  Byte Pair Encoding (BPE)
[30-60 个紧凑 token]   (比传统 bin 化压缩 10×)
```

每个 token 占用 PaliGemma 词表里的"最后 128 个 special token 槽位"。

#### 与 π₀ 对比
| 维度 | π₀（flow matching） | π₀-FAST（AR + FAST） |
|---|---|---|
| 训练速度 | 慢 | **快 5×** |
| 推理速度 | 快（10 步去噪并行） | 慢（自回归一个个出 token） |
| 语言跟随 | 一般 | **更强**（保持 VLM 原生 token 接口） |
| Dexterity | 高 | 同等 |
| 适合场景 | 部署阶段（实时性高） | 训练阶段、对语言敏感的任务 |

#### 著名应用：π₀-FAST-DROID
- 在 **DROID 数据集**（开源、跨大学采集）上 fine-tune 出来的"通用 DROID Franka 策略"
- **零样本部署到 UC Berkeley / Stanford / UW 三个未见过的实验室**都能跑
- openpi 仓库直接提供这个 checkpoint，可以下载就用

#### checkpoint
- `gs://openpi-assets/checkpoints/pi0_fast_base` —— FAST base
- `gs://openpi-assets/checkpoints/pi0_fast_droid` —— DROID 上 fine-tune 完的开箱即用 expert

---

### 4.3 Hi Robot（2025-02）—— 分层 VLA = System 2 + System 1

**博客**：[pi.website/research/hirobot](https://www.physicalintelligence.company/research/hirobot)
**arXiv**：2502.19417

#### 思路（用 Kahneman 的双过程理论类比）
- **System 1（直觉、快速、自动）** ← π₀ 低层策略，处理"已经练过 100 次"的动作
- **System 2（理性、慢速、显式推理）** ← 上层 VLM，内心独白拆解复杂任务

#### 典型例子
用户说："给我做一个素食三明治"

- 上层 VLM（System 2）"想"出来子任务序列：
  1. 找到面包
  2. 抹蛋黄酱
  3. 放生菜（不要肉！素食的）
  4. 盖上另一片面包
- 每一步把子任务作为 prompt 传给底层 π₀（System 1），后者执行具体动作

#### 关键能力
- 解析复杂、多步、带反馈的指令（"那个我不想要"、"小心烧糊了"）
- 跨平台验证：单臂、双臂、移动双臂三种机器人

---

### 4.4 π₀.₅（2025-04）—— 跨家庭开放世界泛化

**论文/博客**：[pi.website/blog/pi05](https://www.physicalintelligence.company/blog/pi05)

#### 核心创新：异质数据 Co-training（联合训练）
π₀.₅ 训练混合了 5 类数据：

```
1. ME — Multiple-Environment 静态机器人在很多不同家庭的数据
2. CE — Cross-Embodiment 来自 π₀ 时期的 8 种机器人数据
3. MM — Mobile manipulation 自家移动机器人在真实家庭的数据
4. HL — High-level 语义子任务标注（图 → "拿起枕头"）
5. WD — Web Data 通用 VLM 数据（VQA、caption、object detection）
```

#### 推理时也是分层的
1. 看到图 + 高级任务（"打扫卧室"）
2. 模型先**自回归预测下一个子任务** → "pick up the pillow"（这一步类似 chain-of-thought）
3. 然后基于子任务 + 图 + state，**走 action expert 用 flow matching 出动作 chunk**

> 这里 π₀.₅ 在结构上是把 "用 VLM 输出文本子任务" 和 "用 action expert 输出动作" 融合到一个网络里完成的。

#### 性能（消融实验）
| 设置 | 训练分布内成功率 | 全新家庭 OOD 成功率 |
|---|---|---|
| 完整 π₀.₅ | 83% | **94%** |
| 去掉 Web Data | 82% | 74% |
| 去掉 CE 数据 | 67% | 49% |
| 去掉 ME 数据 | 57% | 31% |

**结论**：跨机器人数据 (CE+ME) 决定基础性能，互联网数据 (WD) 决定 OOD 物体识别能力。

#### checkpoint
- `gs://openpi-assets/checkpoints/pi05_base`
- `gs://openpi-assets/checkpoints/pi05_libero` —— LIBERO benchmark SOTA（4 个 suite 平均 96.85）
- `gs://openpi-assets/checkpoints/pi05_droid` —— DROID 上 fine-tune

---

### 4.5 Knowledge Insulation（KI，2025-05）

**博客**：[pi.website/research/knowledge_insulation](https://www.physicalintelligence.company/research/knowledge_insulation)

#### 解决的问题
π₀ 直接把 action expert "嫁接"到 PaliGemma backbone 上，让动作专家的梯度直接灌进 VLM。**结果是 VLM 被"污染"**——语言理解能力变弱（让它"把勺子放进盘子"，它去抓了垃圾）。

#### KI 方案
```
┌─────────────────────────────────────────────────────┐
│           VLM Backbone (PaliGemma 3B)               │
│  ┌─────────────────┐    ┌──────────────────────┐    │
│  │ FAST token loss │    │ co-training:          │    │
│  │ (反向传播)       │    │ VQA / caption / OD    │    │
│  └─────────────────┘    └──────────────────────┘    │
└───────────────────┬─────────────────────────────────┘
                    │
              ╳ STOP GRADIENT ╳   ← KI 核心：截断梯度
                    │
              ┌─────▼──────┐
              │ Action Expert │
              │   (300M)   │
              │ flow matching │ ← 推理时用它出动作
              └────────────┘
```

具体做法：
1. VLM backbone 同时学 **FAST 离散动作 token 预测**（让它学到"动作相关的表示"）+ **网页数据**（保留通用知识）
2. Action expert 学 **flow matching 连续动作**，但梯度**不**回传到 VLM
3. 推理时丢掉离散 token 头，只用 flow matching 出动作

#### 实测收益
| 模型 | shirt-folding 成功率 | items-in-drawer 完成率 |
|---|---|---|
| π₀ | ~25% | ~30% |
| π₀-FAST | ~37% | ~40% |
| joint training (污染) | ~50% | ~55% |
| **π₀.₅ + KI** | **~60%** | **~85%** |

> openpi 当前 `pi05` config 默认启用了 KI 训练方式。
> 源码体现：`Pi0Config(pi05=True)` 时 `adarms=True`、`discrete_state_input=True`、loss 路径走 `LiberoOutputs` 的 `discrete_state_input` 分支。

---

### 4.6 Real-Time Chunking（RTC，2025-06）

**博客**：[pi.website/research/real_time_chunking](https://www.physicalintelligence.company/research/real_time_chunking)

#### 解决的问题
π₀ 默认每秒推理一次（50 步 chunk）。但 VLM 推理需要 100-300ms，**这段时间机器人在"等下一段动作"，会出现停顿、抖动甚至切片之间的不连续**。

#### RTC 算法（推理时算法，零训练改动）
把"实时生成下一段 chunk"建模成 **inpainting 问题**：
- 当前 chunk 还剩 K 步没执行（已知未来的"绿色"部分）
- 模型生成新 chunk 时，把这 K 步当作**已知约束**填回去
- 用 flow matching 的去噪过程"inpaint"出剩下的新动作

#### 效果
- 抗延迟：**+200ms 注入延迟，性能几乎不掉**
- 完成动态精细任务：擦火柴、插网线
- 适用：任何 diffusion / flow-based VLA（π₀、π₀.₅）
- 训练时版本（RTC 2.0）后来用在了 π*₀.₆ 的咖啡演示里

---

### 4.7 π₀.₆ 和 π*₀.₆（2025-11）—— 当前最强（闭源）

**博客**：[pi.website/blog/pistar06](https://www.physicalintelligence.company/blog/pistar06)
**Model Card**：[PI06_model_card.pdf](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)

#### π₀.₆（base 模型）相对 π₀.₅ 的改动
1. **VLM backbone 升级**：PaliGemma 3B → **Gemma3 4B**
2. **Action expert 扩大**：300M → **~860M**（与主干同层数）
3. **图像分辨率**：224 → **448**
4. **支持最多 4 路相机**（base + 2 wrist + 可选后向移动底盘相机）
5. **Attention**：图像 token 之间双向，文字 token causal，动作 token 双向
6. **KI 推广**：backbone 同时预测 FAST 离散 token + 多模态 web 数据；action expert 走 flow matching 且 stop-grad

#### π*₀.₆（RL 强化版） —— **Recap 训练方法**
Recap = **R**L with **E**xperience & **C**orrections via **A**dvantage-conditioned **P**olicies

```
完整训练流水线：
1. 模仿学习（IL）打底  ← π₀.₆ base
2. 专家"接管"纠错（Correction phase）
   ─ 让 π₀.₆ 跑起来，专家在出错时遥操作接管
   ─ 把"接管动作"作为高质量监督
3. 真实世界 RL（Reinforcement phase）
   ─ 机器人独立练习，按 episode 结果学好坏
   ─ Advantage-conditioned policy 解决信用分配
```

#### 业务级演示成果
- **咖啡店**：5:30am - 11:30pm 一整天连续做意式咖啡（含磨豆、压粉、扣把、上头、奶泡）
- **洗衣**：50 件**新家、新衣物**无中断折叠
- **箱子组装**：实际包装巧克力的工厂线上组装 + 贴标 59 个
- **吞吐量翻倍 + 失败率减半**

#### 是否开源？
**π₀.₆ 与 π*₀.₆ 当前未在 openpi 仓库放权重。** 这是 π 公司的商业核心，预计也不会近期开源。

---

## 5. openpi 仓库结构（源码导航）

```
openpi/
├── src/openpi/                  ← 核心包
│   ├── models/                  ← 模型定义（JAX/Flax NNX）
│   │   ├── pi0.py                  π₀ / π₀.₅ 共享模型类
│   │   ├── pi0_config.py           Pi0Config（pi05=True 开关）
│   │   ├── pi0_fast.py             π₀-FAST 模型 + Pi0FASTConfig
│   │   ├── gemma.py                双塔 Gemma backbone（含 LoRA）
│   │   ├── gemma_fast.py           AR 解码版本
│   │   ├── siglip.py               SigLIP ViT (So400m/14)
│   │   ├── tokenizer.py            PaliGemma SP + FAST tokenizer
│   │   ├── lora.py                 LoRA 模块
│   │   └── model.py                Observation / Actions 数据结构
│   ├── models_pytorch/          ← 同模型的 PyTorch 实现（2025-09 加入）
│   │   ├── pi0_pytorch.py
│   │   └── transformers_replace/   覆盖 transformers 库的补丁（AdaRMS 等）
│   ├── policies/                ← 策略封装（推理时使用）
│   │   ├── policy.py               Policy 类（JAX/PyTorch 自适应）
│   │   ├── policy_config.py        create_trained_policy 入口
│   │   ├── aloha_policy.py         ALOHA 输入/输出转换
│   │   ├── droid_policy.py         DROID 输入/输出转换
│   │   └── libero_policy.py        LIBERO 输入/输出转换
│   ├── training/                ← 训练相关
│   │   ├── config.py               🌟所有 TrainConfig（含 30+ 预设 config）
│   │   ├── data_loader.py          LeRobot/RLDS 数据加载
│   │   ├── droid_rlds_dataset.py   DROID 大规模 RLDS 加载
│   │   ├── checkpoints.py          Orbax 检查点
│   │   ├── optimizer.py            AdamW + Cosine schedule
│   │   ├── sharding.py             FSDP 分片
│   │   └── weight_loaders.py       从 GCS 加载预训练权重
│   ├── serving/
│   │   └── websocket_policy_server.py    WebSocket 推理服务
│   ├── transforms.py            ← 数据变换 Group/Pipeline
│   └── shared/
│       ├── download.py             GCS gs://openpi-assets 下载
│       ├── normalize.py            归一化 stats
│       └── image_tools.py
│
├── scripts/                     ← 命令行入口
│   ├── train.py                    JAX 单/多卡训练
│   ├── train_pytorch.py            PyTorch DDP 训练
│   ├── compute_norm_stats.py       计算 state/action 归一化均值方差
│   └── serve_policy.py             启动 WebSocket 策略服务
│
├── packages/openpi-client/      ← 客户端轻量库（机器人端 pip install）
│   └── src/openpi_client/
│       ├── websocket_client_policy.py    机器人端 client
│       ├── image_tools.py                resize_with_pad 等
│       ├── action_chunk_broker.py        chunk 缓冲/调度
│       └── runtime/                      可选 runtime 抽象
│
├── examples/                    ← 各机器人平台示例
│   ├── libero/                     LIBERO 仿真 benchmark
│   ├── aloha_sim/                  ALOHA mujoco 仿真
│   ├── aloha_real/                 ALOHA 真机（ROS）
│   ├── droid/                      DROID Franka 真机/仿真
│   ├── ur5/                        UR5 机器人
│   ├── simple_client/              无机器人的随机观测测试
│   ├── convert_jax_model_to_pytorch.py
│   └── inference.ipynb
│
├── third_party/                 ← Git submodule
│   ├── libero/                     LIBERO benchmark
│   └── aloha/                      ALOHA fork（Realsense 改动）
│
├── docs/
│   ├── docker.md
│   ├── norm_stats.md               🌟归一化复用指南（很重要）
│   └── remote_inference.md         🌟远程推理指南
│
├── pyproject.toml               uv 管理依赖
├── README.md
└── LICENSE / LICENSE_GEMMA.txt
```

---

## 6. 完整技术栈

### 6.1 训练侧（JAX 主线）
| 类别 | 选型 | 备注 |
|---|---|---|
| 数值后端 | **JAX 0.5.3 + CUDA 12** | 主开发路径 |
| NN 框架 | **Flax NNX 0.10.2** | Google 新一代 Flax API |
| 优化 / 加速 | `optax`, `chex`, `equinox` | |
| 类型检查 | `jaxtyping`, `beartype` | 数组形状静态检查 |
| 分布式 | JAX FSDP（`fsdp_devices` config） | 单节点多卡，**目前不支持多节点** |
| 数据加载 | `lerobot`（HuggingFace 数据集格式） + `tensorflow_datasets`（RLDS for DROID） | |
| Checkpoint | **`orbax-checkpoint 0.11.13`** | Google 的 ckpt 库 |
| 实验追踪 | `wandb` | |
| CLI | `tyro` | dataclass → CLI 自动生成 |
| 日志 | `tqdm_loggable`, `rich` | |

### 6.2 训练侧（PyTorch，2025-09 加入）
- `torch==2.7.1`
- `transformers==4.53.2`（被覆盖补丁——为了 AdaRMS、精度控制、KV-cache 不更新）
- 通过 `torchrun --standalone` 支持单/多机 DDP
- 注意：**目前不支持 LoRA / FSDP / mixed precision / EMA / π₀-FAST**

### 6.3 模型组件
| 组件 | 选型 |
|---|---|
| 视觉编码器 | **SigLIP So400m/14**（来自 Google big_vision） |
| 语言主干 | **PaliGemma 3B**（Gemma 2B + SigLIP，width=2048, depth=18） |
| Action expert | **gemma_300m**（width=1024, depth=18） |
| Action 生成 | **Flow matching**（π₀/π₀.₅）或 **autoregressive FAST tokens**（π₀-FAST） |
| Tokenizer | SentencePiece（PaliGemma）+ FAST（DCT+BPE，HF AutoProcessor）|
| LoRA 支持 | 自实现，仅 JAX 路径 |

### 6.4 部署侧
- **WebSocket 协议**：`websocket_policy_server.py` + `websocket_client_policy.py`
- **MsgPack-Numpy** 二进制序列化（节省带宽）
- **架构**：策略服务跑在远程 GPU 机器，机器人端只跑轻量 client（`openpi-client` 包，依赖少）
- 推理时机器人端只需把 224×224 uint8 图 + state + prompt 发过去，返回 chunk

### 6.5 数据生态
| 数据集 | 类型 | openpi 中的位置 |
|---|---|---|
| **LeRobot dataset** | π 公司主用格式（HuggingFace 维护） | 默认数据加载格式 |
| **DROID** | 76k+ trajectory，Franka，多大学采集 | `examples/droid/` + RLDS loader |
| **LIBERO** | 130+ task 的仿真 benchmark | `examples/libero/` + Mujoco |
| **ALOHA** | 双臂遥操作演示 | `examples/aloha_*/` + ROS |
| **OXE (Open X-Embodiment)** | 跨多机器人开源集合 | π₀ 预训练里有 |
| 自有内部数据 | π 公司 8+ 机器人 10000+ 小时 | **不开源**（只有 trained weights）|

---

## 7. 端到端工作流（从 0 到部署）

### 7.1 下载预训练 base 直接推理（最简单）

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
    "observation/exterior_image_1_left": ...,  # H,W,3 uint8 或 float
    "observation/wrist_image_left": ...,
    "observation/joint_position": ...,
    "observation/gripper_position": ...,
    "prompt": "pick up the fork",
}
action_chunk = policy.infer(example)["actions"]   # shape (15, 8)
```

### 7.2 在自己的数据上 fine-tune（典型流程）

```bash
# 1. 把数据转换成 LeRobot 格式（每个机器人平台都有示例脚本）
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /your/raw/data

# 2. 计算归一化 stats（state / action 的均值方差）
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 3. 启动训练（JAX 默认）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
    --exp-name=my_experiment --overwrite

# 或 PyTorch
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_libero --exp_name my_experiment

# 4. 起策略服务
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# 5. 机器人端用 openpi-client 连接 WebSocket（端口 8000）
```

### 7.3 数据 pipeline 三层 transform

openpi 的数据流非常工程化，每条样本要过三组 transform：

```
原始 LeRobot dataset
    │
    ▼  ① repack_transforms
       规整字段名（"observation.images.cam_high" → "images.cam_high"）
    │
    ▼  ② data_transforms (robot-specific)
       机器人坐标系、关节顺序映射；action_dim padding 到 32
       例：AlohaInputs / DroidInputs / LiberoInputs（src/openpi/policies/*_policy.py）
    │
    ▼  ③ normalize (norm_stats)
       (x - mean) / std；或 quantile normalization
    │
    ▼  ④ model_transforms (per model type)
       tokenize prompt（PaliGemma SP 或 FAST），构建 attention mask
    │
    ▼
  Observation/Actions 喂模型
```

输出方向：模型出 → 反向 transform → 真实机器人控制指令。

### 7.4 fine-tune 模式与显存

| 模式 | 显存需求 | 适合显卡 | openpi config 示例 |
|---|---|---|---|
| Inference | >8 GB | RTX 4090 | 任何 `pi0_xxx` config |
| LoRA fine-tune | >22.5 GB | RTX 4090 | `pi0_libero_low_mem_finetune` |
| Full fine-tune | >70 GB | A100/H100 80GB | `pi0_libero`, `pi05_libero` |

注：`fsdp_devices > 1` 可以把模型分片到多卡降低单卡显存。

---

## 8. checkpoint 矩阵（你最关心的可用模型清单）

### 8.1 Base 模型（供 fine-tune）

| 名称 | 路径 | 用途 |
|---|---|---|
| π₀ base | `gs://openpi-assets/checkpoints/pi0_base` | 通用 fine-tune 起点 |
| π₀-FAST base | `gs://openpi-assets/checkpoints/pi0_fast_base` | AR 风格 fine-tune |
| π₀.₅ base | `gs://openpi-assets/checkpoints/pi05_base` | 最新版 fine-tune 起点 |

### 8.2 Expert 模型（特定机器人直接用）

| 名称 | 平台 | 路径 | 描述 |
|---|---|---|---|
| π₀-FAST-DROID | Franka DROID | `gs://openpi-assets/checkpoints/pi0_fast_droid` | **跨实验室零样本工作！** |
| π₀-DROID | Franka DROID | `gs://openpi-assets/checkpoints/pi0_droid` | 更快推理，语言跟随弱一些 |
| π₀.₅-DROID | Franka DROID | `gs://openpi-assets/checkpoints/pi05_droid` | 快 + 好语言跟随 |
| π₀-ALOHA-towel | ALOHA | `gs://openpi-assets/checkpoints/pi0_aloha_towel` | 折毛巾 |
| π₀-ALOHA-tupperware | ALOHA | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` | 从保鲜盒拿食物 |
| π₀-ALOHA-pen-uncap | ALOHA | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap` | 开笔盖 |
| π₀.₅-LIBERO | LIBERO 仿真 | `gs://openpi-assets/checkpoints/pi05_libero` | LIBERO benchmark SOTA |

### 8.3 支持的机器人平台（带预训练 norm stats）

| asset_id | 机器人 | 描述 |
|---|---|---|
| `trossen` | ALOHA | 6 自由度双臂 + 平行夹爪 |
| `trossen_mobile` | Mobile ALOHA | ALOHA + Slate 移动底盘 |
| `droid` | Franka (DROID) | 7-DoF + 平行夹爪 |
| `franka` | Franka FR3 | 7-DoF + Robotiq 2F-85 |
| `ur5e` | UR5e | 6-DoF + Robotiq 2F-85 |
| `ur5e_dual` | 双 UR5e | 双臂 |
| `arx` | ARX-5 | 双臂 + 平行夹爪 |
| `arx_mobile` | Mobile ARX-5 | ARX-5 + Slate 底盘 |
| `fibocom_mobile` | Fibocom + 2× ARX-5 | 自有移动机器人 |

### 8.4 标准动作空间约定（必看，会影响 fine-tune）
```
dim 0..5 : 左臂 joint angles      (弧度)
dim 6    : 左臂 gripper          (0=完全张开 → 1=完全闭合)
dim 7..12: 右臂 joint angles     (仅双臂)
dim 13   : 右臂 gripper          (仅双臂)
dim 14..15: 移动底盘 x-y 速度    (仅移动机器人)
```
proprioceptive state 同上但**不含底盘 x-y**（机器人不知道自己绝对位置）。

---

## 9. 关键工程亮点（值得 VLA 研究者抄作业的设计）

1. **Mixture-of-Experts 风格的双 transformer 塔**
   一个 attention block 里有两套 Q/K/V/MLP 权重，token 类型决定走哪套，实现 VLM 与 Action expert 的"软隔离"。这是 π 公司最核心的架构发明。

2. **flow matching 而不是 diffusion**
   损失函数更简单（vector field MSE），不需要噪声 schedule，10 步采样足够。

3. **Action chunking + action horizon = 50**
   不再每步 inference 一次，而是预测整个 1 秒动作 chunk，配合 RTC 实现"边走边想"。

4. **跨平台统一 32 维 action**
   不论 6 / 7 / 14 / 16 DoF 机器人，统一 padding 到 32 维，norm stats 按 asset_id 切换，**单个权重控制所有平台**。

5. **远程推理架构**
   策略跑在 GPU server，机器人端只有 WebSocket client，避免 "GPU + ROS + 机器人驱动" 的依赖地狱。

6. **PaliGemma 而非 Llama/Qwen**
   PaliGemma 视觉编码器 + LLM 一体化、预训练数据更"视觉接地"（grounding）、参数量适中（3B），适合机器人这种端侧低延迟场景。

7. **Knowledge Insulation**
   stop-gradient + FAST aux loss + multi-modal co-training，确保 VLM 主干不被电机控制信号"污染"，是 2025 年最重要的 VLA 训练 trick 之一。

8. **LeRobot 作为 lingua franca**
   π 公司选择 HuggingFace 的 LeRobot 格式作为数据交换格式，让社区贡献数据更顺畅。

---

## 10. 关键术语小白注解（VLA 入门词典）

| 术语 | 含义 |
|---|---|
| **VLA** | Vision-Language-Action 模型 |
| **VLM** | Vision-Language 模型（GPT-4V、Gemini、PaliGemma 这种） |
| **Embodiment** | 具身性。"Cross-embodiment"=跨机器人形态学（双臂/单臂/移动） |
| **Action chunk** | 一次输出的"动作块"（多步动作），常 50 步 |
| **Action horizon** | chunk 的长度 |
| **Action dim** | 单步动作的维度，openpi 统一 32 |
| **Action expert** | π₀ 里专门处理动作的小 transformer（300M） |
| **Flow matching** | 一种连续生成模型，类似 diffusion 但训练更简单 |
| **Tokenizer** | 把连续动作变离散 token 的算法（如 FAST） |
| **adaRMSNorm** | Adaptive RMSNorm，π₀.₅ 用来注入时间步条件 |
| **Knowledge Insulation (KI)** | 训练时截断动作专家→VLM 的梯度，保护 VLM 知识 |
| **Real-Time Chunking (RTC)** | 推理时把"未完成的 chunk"作为 inpaint 约束 |
| **FAST** | Frequency-domain Action Sequence Tokenization（DCT+BPE 压缩动作） |
| **Co-training** | 把机器人数据 + 网页 VLM 数据 + 高级子任务数据混合训练 |
| **DROID** | 大学联合采集的开源 Franka 数据集 |
| **ALOHA** | 斯坦福开源的双臂遥操作机器人 |
| **LIBERO** | 终身学习机器人 benchmark（仿真） |
| **OXE** | Open X-Embodiment dataset（跨形态学开源集） |
| **LeRobot** | HuggingFace 的机器人数据集 / 训练库 |
| **Norm stats** | state/action 的归一化统计量，每个 asset_id 一份 |
| **Asset id** | 机器人平台标识（trossen / droid / ur5e 等） |
| **Repack / data / model transforms** | openpi 三层数据变换管线 |
| **Recap** | π*₀.₆ 的训练算法：演示 + 纠错 + RL 三阶段 |
| **System 1 / 2** | Kahneman 双过程：直觉/反射 vs. 推理/慢思考；Hi Robot 套用此分层 |
| **Open-world generalization** | 在训练集中没见过的家庭/物体上也能工作 |

---

## 11. 学习路径建议（小白如何入门）

### 入门级（理解原理）
1. 读这份文档（你正在读）
2. 看 [π₀ 博客](https://www.physicalintelligence.company/blog/pi0) + [π₀.₅ 博客](https://www.physicalintelligence.company/blog/pi05) 视频
3. 跑 `examples/inference.ipynb` 用预训练 checkpoint 推理一个 dummy observation

### 中级（用 openpi 训自己的模型）
1. 跑通 LIBERO 仿真：`docker compose -f examples/libero/compose.yml up`
2. 阅读 `src/openpi/training/config.py` 里的 `pi05_libero` config，理解每个字段
3. 把你自己的机器人数据 convert 成 LeRobot 格式
4. 写一个 `XxxInputs / XxxOutputs` 类（参考 `libero_policy.py`）
5. 加一个 `TrainConfig`，跑 `compute_norm_stats.py` + `train.py`

### 高级（魔改 / 研究）
1. 精读 `src/openpi/models/pi0.py`（不到 280 行）+ `gemma.py` 双塔实现
2. 读 π₀ 原论文 (arXiv:2410.24164) 和 KI 论文
3. 实验：换 backbone（Llama-3.2-3B？）、换 action 生成方式、加 RL
4. 关注 π 公司新博客 / arXiv（基本每 1-2 个月一篇大新闻）

---

## 12. 已知坑与限制

| 坑 | 说明 |
|---|---|
| 仅 Ubuntu 22.04 测试通过 | 其他系统自行折腾，推荐用 Docker |
| 仅支持 NVIDIA GPU | JAX CUDA 12，AMD/Apple Silicon 不行 |
| 单节点训练 | 多节点目前不支持（torchrun 多节点除外） |
| PyTorch 路径限制多 | 不支持 LoRA / FSDP / mixed-precision / EMA / π₀-FAST |
| transformers 库被打补丁 | PyTorch 路径会用 `cp -r` 覆盖 transformers，**会污染 uv cache** |
| 零样本部署到自家机器人成功率不保证 | "实验性项目"，期望 fine-tune |
| π₀.₆ / π*₀.₆ 不开源 | 商业核心，公司只放 demo |
| 数据格式：必须 LeRobot 或 RLDS | 其他格式需要自己转换 |

---

## 13. 资源汇总（一站式收藏夹）

### 官方
- 公司主页：<https://www.physicalintelligence.company/>
- 简称域名：<https://www.pi.website/>
- GitHub：<https://github.com/Physical-Intelligence/openpi>
- 模型权重：`gs://openpi-assets/checkpoints/` (Google Cloud Storage，匿名可下)
- FAST tokenizer：<https://huggingface.co/physical-intelligence/fast>
- 数据集示例：<https://huggingface.co/datasets/physical-intelligence/libero>

### 论文
- π₀：<https://arxiv.org/abs/2410.24164>
- π₀-FAST：blog 内有 PDF 链接
- Hi Robot：<https://arxiv.org/abs/2502.19417>
- π₀.₅：博客内 paper
- Knowledge Insulation：<https://www.physicalintelligence.company/research/knowledge_insulation>
- RTC：博客内 paper
- π*₀.₆：博客内 paper + [Model Card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)

### 关联生态
- LeRobot（HuggingFace）：<https://github.com/huggingface/lerobot>
- DROID：<https://droid-dataset.github.io/>
- LIBERO：<https://libero-project.github.io/>
- ALOHA：<https://tonyzhaozh.github.io/aloha/>
- PaliGemma：<https://ai.google.dev/gemma/docs/paligemma>
- Open X-Embodiment：<https://robotics-transformer-x.github.io/>

---

## 14. 一句话总结（送给小白）

> π 公司的核心配方 = **"PaliGemma 视觉语言模型主干"** + **"专门的动作专家小模型"** + **"flow matching 生成连续动作"** + **"跨机器人统一 32 维动作空间"** + **"Knowledge Insulation 保护 VLM 知识"** + **"Real-Time Chunking 解决延迟"** + **"Recap RL 让模仿学习走向工业级可靠"**。

openpi 是这套配方的开源参考实现，你完全可以基于 `pi05_base` 在自己的双臂机器人上 fine-tune，30000 步训练（A100 一天左右）就能拿到一个能用的 demo 策略。

---

*文档版本：2026-05*
*维护者：基于 openpi @ HEAD（2025-09 PyTorch 支持版本）+ pi.website 全部公开内容*
