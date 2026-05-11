# 03 · 技术栈逐项解读

> 解释 openpi 每个依赖**为什么选这个**、**版本兼容矩阵**、**遇到坑怎么办**。
>
> 想 fork / 升级依赖前必读。

---

## 1. 一张图看全栈

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            机器人 client（轻量）                          │
│   openpi-client + websockets + msgpack-numpy                              │
└───────────────────┬───────────────────────────────────────────────────────┘
                    │ WebSocket / msgpack 双向
┌───────────────────▼───────────────────────────────────────────────────────┐
│                              policy server                                │
│   ┌────────────────────────────────────────────────────────────────────┐ │
│   │  推理 / 训练（JAX 路径）                                            │ │
│   │   ├─ jax 0.5.3 + jaxtyping + beartype（运行时 shape 检查）          │ │
│   │   ├─ flax 0.10.2（NNX 风格 + Linen bridge）                         │ │
│   │   ├─ optax（AdamW + cosine/rsqrt schedule）                         │ │
│   │   ├─ orbax-checkpoint 0.11.13（异步增量 ckpt + GCS sink）          │ │
│   │   ├─ einops（einops.rearrange）                                     │ │
│   │   ├─ etils.epath（统一本地/GCS path）                               │ │
│   │   └─ wandb（日志）                                                   │ │
│   │  推理 / 训练（PyTorch 路径）                                        │ │
│   │   ├─ torch 2.7.1 + transformers 4.53.2（带定制 attention）          │ │
│   │   └─ FSDP / DDP                                                      │ │
│   ├─ 数据                                                                  │
│   │   ├─ lerobot（HF datasets + LeRobot 格式 .arrow shard）              │ │
│   │   ├─ tensorflow-cpu 2.15 + tensorflow-datasets（DROID RLDS）         │ │
│   │   ├─ dlimp（高速 RLDS → tf.data 管线）                                │ │
│   │   ├─ sentencepiece（PaliGemma 分词器）                                │ │
│   │   ├─ transformers（FAST tokenizer 走 AutoProcessor）                  │ │
│   │   ├─ augmax（JAX 上的图像增强）                                       │ │
│   │   ├─ opencv-python, pillow, imageio（图像 IO）                        │ │
│   │   └─ polars（轻量 DataFrame）                                          │ │
│   └─ 部署                                                                  │
│       ├─ websockets（asyncio）                                            │ │
│       ├─ msgpack-numpy（高效 numpy 序列化）                               │ │
│       └─ Docker（多镜像：aloha_sim/libero/droid 等）                       │ │
└───────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│        权重 / 数据存储：Google Cloud Storage (gs://openpi-assets/...)     │
│        本地 cache: ~/.cache/openpi、HF datasets cache                     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 依赖矩阵（pyproject.toml 核心字段）

> 来源：`references/openpi/pyproject.toml:8-41`

| 类别            | 包                            | 钉版          | 为什么选 / 注意点                                                                                            |
| --------------- | ----------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------ |
| **JAX 核心**    | `jax[cuda12]`                 | `==0.5.3`     | 钉死 0.5.3 是因为 NNX bridge 和 Orbax 兼容；CUDA 12 → 默认匹配 H100 / RTX 4090 / A100                       |
|                 | `flax`                        | `==0.10.2`    | NNX 风格 + Linen bridge（`flax.nnx.bridge.ToNNX`，pi0.py:73）。NNX 还未完全成熟，所以一些子模块仍 Linen     |
|                 | `jaxtyping`                   | `==0.2.36`    | 给 `at.Float[at.Array, "b t d"]` 做运行时 shape 检查                                                         |
|                 | `chex`                        | `==0.1.90`    | jax 单测工具                                                                                                 |
|                 | `equinox`                     | `>=0.11.8`    | 备用 PyTree 工具（部分场景）                                                                                 |
|                 | `optax`                       | （间接）      | optimizer，从 jax 生态                                                                                       |
|                 | `orbax-checkpoint`            | `==0.11.13`   | checkpoint：异步保存、增量、GCS sink、PyTree 友好。**钉死，否则 NNX 兼容会坏**                                |
|                 | `ml-dtypes`                   | `==0.4.1`     | jax 内部依赖，被 override 是因为 tf 2.15 也要它                                                              |
|                 | `tensorstore`                 | `==0.1.74`    | orbax 后端                                                                                                   |
|                 | `beartype`                    | `==0.19.0`    | jaxtyping 后端                                                                                               |
| **PyTorch**     | `torch`                       | `==2.7.1`     | PyTorch 路径的训练/推理。`models_pytorch/`                                                                   |
|                 | `transformers`                | `==4.53.2`    | HF；`models_pytorch/transformers_replace/` 里改了 attention/FSDP 兼容                                        |
| **数据**        | `lerobot`                     | git rev pinned| HuggingFace LeRobot：标准化机器人数据格式（episodes/、meta/、stats/）；走 git rev 因为 PyPI 更新慢            |
|                 | `dlimp`                       | git rev pinned| RLDS → tf.data 高速 pipeline，DROID 必用                                                                     |
|                 | `tensorflow-cpu`              | `==2.15.0`    | 只 RLDS 数据加载用 TF，所以 CPU 即可。**注意 cp311 only**                                                    |
|                 | `tensorflow-datasets`         | `==4.9.9`     | RLDS 格式读取                                                                                                |
|                 | `gym-aloha`                   | `>=0.1.1`     | ALOHA 仿真环境（gym 接口）                                                                                   |
|                 | `polars`                      | `>=1.30.0`    | 轻量 DataFrame，比 pandas 快                                                                                 |
|                 | `numpy`                       | `>=1.22.4,<2`| 钉低于 2.0 因为 TF 2.15 不兼容 numpy 2                                                                       |
|                 | `numpydantic`                 | `>=1.6.6`     | numpy + pydantic schema                                                                                      |
| **图像**        | `opencv-python`               | `>=4.10`      | resize / 色彩转换                                                                                            |
|                 | `pillow`                      | `>=11.0`      | 图像 IO                                                                                                      |
|                 | `imageio`                     | `>=2.36`      | 视频读写（可视化）                                                                                           |
|                 | `augmax`                      | `>=0.3.4`     | JAX 端图像增强（被 data pipeline 调用）                                                                      |
| **NLP**         | `sentencepiece`               | `>=0.2.0`     | PaliGemma 分词                                                                                               |
| **推理通信**    | `websockets`                  | （间接）      | server / client                                                                                              |
|                 | `msgpack-numpy`               | （间接）      | numpy → bytes 高效序列化                                                                                     |
|                 | `fsspec[gcs]`                 | `>=2024.6`    | 透明 GCS path 读写                                                                                           |
| **配置 / CLI**  | `tyro`                        | `>=0.9.5`     | 从 dataclass 自动生成 CLI（`scripts/train.py pi0_libero --exp-name=...`）                                    |
|                 | `ml_collections`              | `==1.0.0`     | 嵌套 config（旧 API，部分模块还在用）                                                                        |
|                 | `flatbuffers`                 | `>=24.3`      | tf-datasets 依赖                                                                                             |
| **日志 / UI**   | `wandb`                       | `>=0.19.1`    | 训练日志可视化                                                                                               |
|                 | `tqdm-loggable`               | `>=0.2`       | tqdm 但写到 logger                                                                                           |
|                 | `rich`                        | `>=14.0.0`    | 美化 print                                                                                                   |
|                 | `treescope`                   | `>=0.1.7`     | JAX PyTree 可视化                                                                                            |
| **工具**        | `dm-tree`                     | `>=0.1.8`     | PyTree 操作                                                                                                  |
|                 | `einops`                      | `>=0.8.0`     | 张量 reshape DSL                                                                                             |
|                 | `etils.epath`                 | （间接）      | 统一 `Path("gs://...")` / 本地路径                                                                           |
|                 | `filelock`                    | `>=3.16`      | 多进程下载锁                                                                                                 |
|                 | `typing-extensions`           | `>=4.12`      | Protocol / TypeAlias                                                                                         |
| **本地工具包**  | `openpi-client`               | workspace     | `packages/openpi-client/` 里的本地包：客户端轻量入口                                                         |
| **dev**         | `pytest, ruff, pre-commit, ipykernel, matplotlib, pynvml` | | 开发依赖（`uv sync --group dev`）                                                                            |
| **rlds 可选组**| `dlimp, tensorflow-cpu==2.15.0, tensorflow-datasets==4.9.9` | | DROID 训练才需要：`uv sync --group rlds`                                                                     |

**关键 override**（`pyproject.toml:65`）：
```toml
[tool.uv]
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74"]
```
因为 tf 2.15 + jax 0.5.3 + orbax 0.11 这三者的 `ml-dtypes` 想要不同版本，需要强制对齐。

---

## 3. 硬件 / GPU 要求

来源：`references/openpi/README.md:26-31`

| 模式               | 显存       | 推荐 GPU              | 备注                                                            |
| ------------------ | ---------- | --------------------- | --------------------------------------------------------------- |
| 推理               | > 8 GB     | RTX 4090              | bf16 + KV cache                                                 |
| Fine-Tune（LoRA）  | > 22.5 GB  | RTX 4090              | 只训 LoRA + bf16 冻结主干                                       |
| Fine-Tune（Full）  | > 70 GB    | A100 80GB / H100      | 全量微调；可用多卡 FSDP 分散：`fsdp_devices=8` 可在 8×A100 上跑 |

**测试过的 OS**：Ubuntu 22.04。其他 OS 不保证（macOS 没 CUDA；Windows 缺 tf-cpu）。

---

## 4. JAX 路径 vs PyTorch 路径对比

| 维度                  | JAX 路径（默认）                                      | PyTorch 路径                                            |
| --------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| 模型源码              | `src/openpi/models/*.py`（NNX + Linen bridge）        | `src/openpi/models_pytorch/*.py`                        |
| 训练脚本              | `scripts/train.py`                                    | `scripts/train_pytorch.py`                              |
| 优化器                | optax（AdamW + chain + clip_by_global_norm）          | torch.optim.AdamW + grad clip                           |
| 并行                  | FSDP via `jax.sharding.NamedSharding`（自动）         | PyTorch FSDP / DDP                                      |
| Checkpoint            | Orbax（异步、增量、PyTree）                            | torch save_state                                         |
| Tokenizer             | sentencepiece + transformers.AutoProcessor            | 同上                                                    |
| 推理 KV cache         | 手写 `jax.lax.while_loop`                              | HuggingFace generate（修改过）                          |
| 适合场景              | 大规模训练（TPU/H100）、最低延迟推理                  | 不熟悉 JAX、需要 HF 生态                                |
| 入门难度              | 高（要懂 NNX state 概念）                              | 低（HF 风格）                                           |
| 性能                  | 通常更快（XLA 编译）                                   | 可接受但略慢                                            |

**结论**：研究 / 上线推荐 JAX 路径；快速做 demo / 入门可用 PyTorch 路径。

---

## 5. 依赖版本"地雷"——容易踩的坑

| 坑                                                | 原因                                                  | 解法                                                     |
| ------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| `numpy>=2.0` 装上后 TF 2.15 报错                  | TF 2.15 要 numpy<2                                    | 锁 `numpy<2.0`                                           |
| 升 jax 到 0.5.4+，orbax 报 PyTree 不匹配          | NNX 内部表示更新                                       | 钉 jax 0.5.3 + orbax 0.11.13                             |
| Python 3.12 装 tf-cpu 失败                        | tf 2.15 只给 cp311 wheel                              | `uv venv --python 3.11`                                  |
| `pip install lerobot` 装不到固定 commit           | LeRobot 在快速迭代                                    | 走 `[tool.uv.sources]` 的 git rev pin                    |
| CUDA mismatch（CUDA 11 机器装了 jax cuda12）      | jax\[cuda12] 要 CUDA 12 driver                        | 升级 NVIDIA driver 或换 jax\[cuda11_pip]                  |
| GIT_LFS_SMUDGE 拉 LeRobot 时拉了很多大文件        | LeRobot 仓库带 LFS                                    | `GIT_LFS_SKIP_SMUDGE=1 uv sync`                          |
| 编译耗时长（首次启动 jit）                        | XLA 编译 ~5-10 min                                    | 复用 `~/.cache/jax`，scripts/train.py 已配置             |

---

## 6. 关键库的"心智模型"

### 6.1 JAX
- **函数式**：所有更新返回新对象（`tx.update` 返回新 opt_state，`optax.apply_updates` 返回新 params）。
- **`jit`**：把 Python 函数编译成 XLA HLO；输入输出 shape/dtype 改变就重新编译（缓存到 `~/.cache/jax`）。
- **`pmap` / `shard_map` / `with_sharding_constraint`**：多设备并行的不同抽象层；openpi 用最现代的 `NamedSharding + PartitionSpec`。
- **PRNG**：显式传递 `rng`，`jax.random.split` 派生子 key。

### 6.2 Flax NNX
- NNX 是 Flax 的"PyTorch 风格" API：模块带状态、可变。
- `nnx.split(model) → (graphdef, state)`：分离图结构和参数；`nnx.merge` 合回。
- `nnx.value_and_grad(loss_fn, argnums=DiffState(0, filter))`：选择性求导（用 filter 过滤可训参数）。
- `nnx_bridge.ToNNX(linen_module)`：包裹 Linen 模块成 NNX；openpi 因为 gemma 还没改成 NNX，所以用 bridge。

### 6.3 Orbax
- `CheckpointManager(dir, options)`：管理多版本 ckpt；自动清理旧的、保留 keep_period 间隔的。
- `save({'params': params, 'opt_state': opt_state, 'step': step})`：同步/异步保存（默认异步）。
- 存储格式：tensorstore + zarr，分片到 GCS 友好。

### 6.4 LeRobot
- 数据格式：每条 episode 是个 .arrow 文件；meta 里有 `info.json`（schema）、`stats.json`（每列统计）、`tasks.jsonl`（任务名）。
- `lerobot_dataset.LeRobotDataset(repo_id, ...)`：从 HF hub 拉数据；`__getitem__(idx)` 返回 dict（state, actions, images, ...）。
- 与 openpi 接口：openpi 在 LeRobot 之上加 `repack_transforms`（把 LeRobot 的字段名映射到 openpi 内部约定）。

---

## 7. 选择 / 替换其他组件的可行性

| 想换什么            | 可行性                       | 注意                                                                      |
| ------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| PaliGemma → LLaVA   | 需重写 weight_loader + tokenizer | 整个 prompt 编码、词表对齐要重做                                          |
| SigLIP → DINOv2     | 中等                         | 替换 `src/openpi/models/siglip.py`，确保 patch token 数 / width 匹配 LLM  |
| flow matching → DDPM| 容易                          | 只改 `compute_loss` / `sample_actions`，模型结构不变                      |
| JAX → 纯 PyTorch    | 已支持                       | 用 `train_pytorch.py`，但部分新特性 JAX 先支持                            |
| WebSocket → gRPC    | 容易                          | 改 `serving/`，序列化保持 msgpack 即可                                    |

---

下一篇：[`04-training-pipeline.md`](04-training-pipeline.md) — 训练 pipeline 全流程拆解（数据 → 归一化 → FSDP → 优化器 → 检查点）。
