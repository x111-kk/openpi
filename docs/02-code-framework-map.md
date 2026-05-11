# 02 · 代码框架地图

> 目的：把 openpi 仓库的每一个目录、每一个 .py 文件按职责分类，并给出**关键类/函数 + 行号**锚点。
>
> 把这一篇打开放在另一个屏幕，对着 IDE 跳转源码读，效率最高。

---

## 0. 顶层目录速览

```
openpi/                                  # ← submodule 位置：references/openpi/
├── README.md                            # 官方安装 / 模型矩阵
├── pyproject.toml                       # 依赖（详见 03-tech-stack.md）
├── uv.lock                              # uv 锁文件
├── scripts/                             # 命令行入口
│   ├── train.py                         # JAX 训练入口
│   ├── train_pytorch.py                 # PyTorch 训练入口
│   ├── compute_norm_stats.py            # 计算数据归一化统计
│   ├── serve_policy.py                  # WebSocket policy server
│   └── docker/                          # Docker 镜像
├── src/openpi/                          # 主代码包
│   ├── models/                          # 模型定义（JAX）
│   ├── models_pytorch/                  # 模型定义（PyTorch 等价实现）
│   ├── policies/                        # Policy 推理封装 + 各机器人的 input/output transforms
│   ├── serving/                         # WebSocket 远程推理 server/client
│   ├── shared/                          # 工具（normalize, download, type checks）
│   ├── training/                        # 训练 pipeline（data loader, optimizer, sharding, ckpt）
│   └── transforms.py                    # 数据 transform 基础设施（详见 06-data-transforms.md）
├── packages/openpi-client/              # 客户端轻量包（远程推理时机器人端用）
├── examples/                            # 真机/仿真集成示例（aloha_real, droid, libero, simple_client）
├── docs/                                # 官方英文文档
└── third_party/                         # 第三方代码（如 gemma 原版参考）
```

---

## 1. `src/openpi/models/` —— 模型定义（JAX）

| 文件                  | 行数 | 作用                                                                                                                       |
| --------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------- |
| `model.py`            | 332  | `BaseModel`, `BaseModelConfig`, `Observation`, `Actions`, `restore_params`, `preprocess_observation`，所有模型的共同基础类 |
| `pi0_config.py`       | 117  | `Pi0Config` dataclass：`action_dim=32, action_horizon=50, max_token_len=48`，paligemma/action expert variant 选择          |
| `pi0.py`              | 279  | `Pi0` 主类：`embed_prefix` (107)、`embed_suffix` (140)、`compute_loss` (189)、`sample_actions` (217)                       |
| `pi0_fast.py`         | 313  | `Pi0FAST` 主类：自回归版本，`embed_inputs` (160)、`compute_loss` (198)、`sample_actions` (236)                             |
| `gemma.py`            | 459  | `Module`, `Block`, `Attention`, `Embedder`, `RMSNorm`(含 AdaRMS), `get_config`, RoPE, GQA                                  |
| `gemma_fast.py`       | 437  | π₀-FAST 用的 Gemma 变体：原生支持 KV cache + 自回归解码                                                                    |
| `siglip.py`           | 373  | SigLIP ViT (So400m/14)：`Module`, `MAPHead`, patch embed + 27 层 transformer                                               |
| `vit.py`              | 307  | ViT 通用实现（被 siglip 使用）                                                                                             |
| `lora.py`             | 148  | `LoRAConfig`, `Einsum`(33), `FeedForward`(88) —— 任意 einsum/FFN 加 LoRA                                                  |
| `tokenizer.py`        | 371  | `PaligemmaTokenizer`(14), `FASTTokenizer`(51), `FSQTokenizer`                                                              |
| `model_test.py`       | 94   | 单测                                                                                                                       |
| `pi0_test.py`         | 46   | 单测                                                                                                                       |
| `lora_test.py`        | 94   | 单测                                                                                                                       |
| `tokenizer_test.py`   | 27   | 单测                                                                                                                       |

**关键锚点**：
- `model.py::Observation` → `images: dict[name, [B,H,W,3]]`, `state: [B, D]`, `tokenized_prompt: [B, L]`, `tokenized_prompt_mask: [B, L]`, `token_ar_mask: [B, L]`（仅 FAST）
- `pi0.py:19 make_attn_mask` → 混合双向/因果 mask 生成
- `pi0.py:48 posemb_sincos` → 时间嵌入
- `gemma.py:300 Block.__call__` → 一层 transformer 完整逻辑
- `gemma.py:359 nn.scan(...)` → 把 N 层 Block 用 scan 折叠，节省编译时间

---

## 2. `src/openpi/models_pytorch/` —— PyTorch 等价实现

| 文件                      | 作用                                                                          |
| ------------------------- | ----------------------------------------------------------------------------- |
| `pi0_pytorch.py`          | `Pi0` 模型 PyTorch 实现                                                       |
| `pi0_fast_pytorch.py`     | `Pi0FAST` PyTorch 实现                                                        |
| `gemma_pytorch.py`        | Gemma transformer PyTorch                                                     |
| `siglip_pytorch.py`       | SigLIP PyTorch                                                                |
| `transformers_replace/`   | 修改过的 HuggingFace transformers 文件（FSDP / 自定义 attention）             |
| `preprocessing_pytorch.py`| `preprocess_observation` PyTorch 版                                           |

PyTorch 路径默认用 HuggingFace transformers + FSDP（DeepSpeed-style），适合不熟悉 JAX 的用户。**`scripts/train_pytorch.py` 是入口**。

---

## 3. `src/openpi/policies/` —— Policy 封装 + 机器人特定 transform

| 文件               | 作用                                                                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `policy.py`        | `Policy`(24) 类：用户 obs dict → transform → batch → `model.sample_actions` → output transform → dict。统一 JAX/PyTorch 接口                              |
| `policy_config.py` | `PolicyConfig`(94) + `create_trained_policy`：从 checkpoint 路径还原 model+config+transforms                                                              |
| `aloha_policy.py`  | `AlohaInputs`(25), `AlohaOutputs`(91)：ALOHA 14-DoF（双臂 7+7）joint+gripper 与 π 内部空间的双向转换                                                       |
| `droid_policy.py`  | `DroidInputs`, `DroidOutputs`：DROID 8-DoF（6-DoF joint vel + gripper）转 32-DoF 统一动作空间                                                              |
| `libero_policy.py` | `LiberoInputs`, `LiberoOutputs`：LIBERO 7-DoF Franka arm                                                                                                   |
| `policy_test.py`   | 单测                                                                                                                                                       |

**关键锚点**：
- `policy.py:67 infer` —— 一次推理的完整流程（transform → batch → JIT sample_actions → unbatch → output transform）
- `policy.py:94 outputs["actions"] = self._sample_actions(rng, observation)` —— 模型实际调用的位置（注意被 `module_jit` 包了一层，无 host 开销）

---

## 4. `src/openpi/serving/` —— 远程推理

| 文件                            | 作用                                                                                                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `websocket_policy_server.py`    | `WebsocketPolicyServer`(15)：协议 = `msgpack_numpy` 序列化 obs/action，每次连接发送一次 metadata，然后 obs→action 循环                                |
| `http_policy_server.py`         | HTTP REST 版本（较少用）                                                                                                                              |
| `__init__.py`                   | 导出                                                                                                                                                  |

**协议示意**：
```
机器人 client ──────[msgpack(obs)]──────▶ server
机器人 client ◀────[msgpack(action)]───── server.infer(obs)
```

`packages/openpi-client/` 是客户端轻量库，机器人端只需安装这个就能用 `WebsocketClientPolicy` 通信。

---

## 5. `src/openpi/training/` —— 训练框架

| 文件                       | 行数 | 作用                                                                                                                                |
| -------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                | 989  | **核心**：`TrainConfig`(466), `DataConfig`(65), `DataConfigFactory` 系列(204+) + 30+ 个预定义 `TrainConfig` 实例                    |
| `data_loader.py`           | 540  | `Dataset` / `IterableDataset` / `DataLoader` 协议；`TransformedDataset`(53), `FakeDataset`(99), `create_data_loader`               |
| `droid_rlds_dataset.py`    | 248  | DROID（200k+ 集，70k+ 小时）的 RLDS（TF reverb-style）数据集加载器                                                                  |
| `sharding.py`              | 102  | `make_mesh`(17), `fsdp_sharding`(48)：自动按 array 大小决定 FSDP 切片                                                              |
| `optimizer.py`             | 109  | `AdamW`(66), `SGD`(89), `CosineDecaySchedule`(16), `RsqrtDecaySchedule`(35)                                                       |
| `weight_loaders.py`        | 104  | `CheckpointWeightLoader`(38) 加载 `gs://openpi-assets/...`; `PaliGemmaWeightLoader`(58) 加载 PaliGemma 官方权重                    |
| `checkpoints.py`           | 159  | Orbax 封装：`initialize_checkpoint_dir`, `save_state`, `restore_state`                                                              |
| `utils.py`                 | 38   | `TrainState` dataclass：`step, params, model_def, tx, opt_state, ema_decay, ema_params`                                            |
| `data_loader_test.py`      | 84   | 单测                                                                                                                                |
| `misc/polaris_config.py`   |      | Polaris 超算专用 config                                                                                                             |
| `misc/roboarena_config.py` |      | RoboArena benchmark config                                                                                                          |

**预定义训练 config 列表**（`config.py:564-977` 的 `_CONFIGS = [...]`）：
```
pi0_aloha
pi0_aloha_pen_uncap
pi0_aloha_sim
pi0_libero
pi0_libero_low_mem_finetune
pi0_fast_libero
pi0_fast_libero_finetune
pi0_droid
pi0_fast_droid
pi05_droid
pi0_base
pi0_fast_base
pi05_base
debug
debug_pi0_fast
... （共 30+ 个）
```
直接 `uv run scripts/train.py <config_name>` 即可启动训练。

---

## 6. `src/openpi/shared/` —— 通用工具

| 文件                  | 作用                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `array_typing.py`     | `at.Float[Array, "b t d"]` 这类 jaxtyping shape 标注 + `typecheck` 装饰器（beartype 后端）                    |
| `download.py`         | `maybe_download(gs://...)` 透明 GCS/HF cache 下载                                                             |
| `normalize.py`        | `NormStats(mean, std, q01, q99)` + 量化归一化 vs zscore                                                       |
| `nnx_utils.py`        | `module_jit`, `PathRegex`(用 regex 过滤 nnx state), `state_map`                                              |
| `image_tools.py`      | 图像处理（resize, color jitter——多在 client 端用）                                                            |

---

## 7. `src/openpi/transforms.py` —— 核心 transform 框架

详见 [`06-data-transforms.md`](06-data-transforms.md)。关键类：

```
transforms.py
├── DataTransformFn (Protocol, 24)        # 任何 transform 必须实现 __call__(data) → data
├── Group (40)                            # (inputs, outputs) 一对 transform，supports push()
├── CompositeTransform (63)               # 多个 transform 串接
├── RepackTransform (80)                  # 用 path 字符串重排嵌套 dict
├── InjectDefaultPrompt (105)             # obs 缺 prompt 时注入默认
├── Normalize (115)                       # 用 NormStats 归一化 / 反归一化
├── Unnormalize                           # 反向（policy 输出后用）
├── ResizeImages
├── TokenizePrompt                        # 用 PaligemmaTokenizer
├── TokenizeFASTInputs                    # 用 FASTTokenizer
├── PromptFromLeRobotTask                 # 从 LeRobot dataset.meta 取语言指令
└── (其他若干工具 transform)
```

---

## 8. `scripts/` —— 命令行入口

| 文件                      | 作用                                                                                            | 典型调用                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `train.py`                | JAX 训练主循环（init → load weights → FSDP → train_step×N → ckpt → wandb）                      | `uv run scripts/train.py pi0_libero --exp-name=my_run`                            |
| `train_pytorch.py`        | PyTorch 训练（HF transformers + FSDP/DDP）                                                      | `uv run scripts/train_pytorch.py pi0_libero --exp-name=...`                       |
| `compute_norm_stats.py`   | 跑一遍数据集，计算 mean/std/quantile 写到 `assets/<asset_id>/norm_stats.json`                   | `uv run scripts/compute_norm_stats.py pi0_libero`                                 |
| `serve_policy.py`         | 启动 WebSocket policy server                                                                    | `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero ...` |
| `docker/`                 | 多种 Docker 镜像 Dockerfile（aloha_sim, libero_sim, droid 等）                                  |                                                                                   |

---

## 9. `examples/` —— 集成示例

| 子目录              | 作用                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `aloha_real/`       | 真机 ALOHA 集成（含 ROS 节点）                                                                |
| `aloha_sim/`        | gym-aloha 仿真集成                                                                            |
| `libero/`           | LIBERO benchmark 推理 + 评估                                                                  |
| `droid/`            | DROID 数据集训练教程 + 推理脚本                                                               |
| `simple_client/`    | 最小 WebSocket client 示例                                                                    |
| `inference.ipynb`   | 最小推理 notebook                                                                             |
| `policy_records.ipynb` | 可视化 Policy 录制                                                                         |

---

## 10. 一次推理走过的所有文件（端到端调用链）

```
机器人 client                                                  policy server
─────────────────                                              ─────────────────
WebsocketClientPolicy.infer(obs)                              websocket_policy_server.py:_handler
        │                                                              │
        │ ──── msgpack.pack({images, state, prompt}) ─────────────────▶│
        │                                                              ▼
        │                                                   Policy.infer(obs)              # policies/policy.py:67
        │                                                              │
        │                                                              ▼  (data transforms)
        │                                                   transforms.compose(...)         # transforms.py:74
        │                                                       ├─ AlohaInputs               # policies/aloha_policy.py:25
        │                                                       ├─ ResizeImages              # transforms.py
        │                                                       ├─ Normalize                 # transforms.py:115
        │                                                       ├─ InjectDefaultPrompt
        │                                                       └─ TokenizePrompt            # 调用 PaligemmaTokenizer
        │                                                              │
        │                                                              ▼  (batch + to JAX)
        │                                                   jax.tree.map(jnp.asarray(...))
        │                                                              │
        │                                                              ▼  (jit-compiled forward)
        │                                                   model.sample_actions(rng, obs)  # models/pi0.py:217
        │                                                       ├─ embed_prefix              # images + language
        │                                                       │      ├─ siglip.Module      # 视觉编码
        │                                                       │      └─ PaliGemma.llm(embed_only)
        │                                                       ├─ llm forward → KV cache    # gemma.py
        │                                                       └─ while τ ≥ 0:              # ODE 10 步
        │                                                            ├─ embed_suffix         # state + noisy_action + time
        │                                                            ├─ llm(...) with cache  # gemma.py:Module
        │                                                            └─ action_out_proj → v_t
        │                                                              │
        │                                                              ▼  (output transforms)
        │                                                   Unnormalize → AlohaOutputs (joint flip / gripper unmap)
        │                                                              │
        │ ◀──── msgpack.pack({actions: [50, 14]}) ─────────────────────│
        ▼
Robot executes action chunk
```

---

下一篇：[`03-tech-stack.md`](03-tech-stack.md) — 每个依赖为什么选这个版本、有什么坑、能不能换。
