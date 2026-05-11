# 04 · 训练 pipeline 全流程拆解

> 从命令行 `uv run scripts/train.py pi0_libero --exp-name=my_run` 开始，把训练循环里发生的每一件事讲清楚。
>
> 涉及：CLI → config → 数据加载 → 归一化 → 权重初始化 → FSDP 切片 → JIT 训练步 → EMA → Orbax 检查点 → WandB 日志。

---

## 1. 训练前置：先算归一化统计

任何新数据集训练前都要先跑一次：
```bash
uv run scripts/compute_norm_stats.py pi0_libero
```

它做了什么？（`scripts/compute_norm_stats.py:89`）
1. 读 `_CONFIGS["pi0_libero"]`，构造 `data_config`
2. 用 `_data_loader.create_torch_dataset(data_config, action_horizon, model_config)` 拉 LeRobot 数据
3. 套上 `repack_transforms` + `data_transforms`（机器人特定 transform），但**不归一化、不 tokenize**
4. 遍历每个 batch，累计 `RunningStats`（Welford 在线算法 + 分位数 ApproxQuantile）
5. 把结果写到 `assets/<repo_id>/norm_stats.json`，下次训练自动读

结构：
```json
{
  "state":   {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
  "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
}
```

`use_quantile_norm=True` 时用 `(q01, q99)` 做 robust z-score（截断离群点）；否则用 `(mean, std)`。

---

## 2. CLI 入口：tyro 把 TrainConfig 变成参数

`scripts/train.py:280` 的 main：
```python
if __name__ == "__main__":
    main(_config.cli())
```

`_config.cli()` (`training/config.py:978`)：
```python
def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in {c.name: c for c in _CONFIGS}.items()})
```

行为：
- `_CONFIGS` 是 30+ 个预定义的 `TrainConfig` 实例（pi0_aloha, pi0_libero, pi05_droid, ...）
- tyro 让你 `python train.py pi0_libero` 选 config，并允许 `--batch-size 256 --num-train-steps 10000` 这种 inline 覆盖

`TrainConfig` 字段（`config.py:466-562`）：
```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str                                    # 唯一 ID
    model: BaseModelConfig                       # Pi0Config / Pi0FASTConfig
    data: DataConfigFactory                      # 数据 factory
    weight_loader: WeightLoader = NoOpWeightLoader()  # 从哪加载初始权重
    lr_schedule: LRScheduleConfig = CosineDecaySchedule()
    optimizer: OptimizerConfig = AdamW()
    ema_decay: float | None = 0.99
    freeze_filter: nnx.filterlib.Filter = nnx.Nothing  # 哪些 param 不训
    num_train_steps: int = 30_000
    batch_size: int = 32
    num_workers: int = 2
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    seed: int = 42
    fsdp_devices: int = 1                        # 多少卡走 FSDP
    project_name: str
    exp_name: str                                # 实验名（必填）
    wandb_enabled: bool = True
    overwrite: bool = False
    resume: bool = False
    # ...
```

---

## 3. main() 全流程（带行号）

`scripts/train.py:194 main(config)`：

```
194 │ main(config)
195 │   init_logging()                            # 自定义 log 格式
196 │   if batch_size % device_count != 0: raise
203 │   jax.config.update("jax_compilation_cache_dir", "~/.cache/jax")
205 │   rng = jax.random.key(seed)
208 │   mesh = sharding.make_mesh(fsdp_devices)   # 见 §6
212 │   checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(...)
218 │   init_wandb(config, resuming=resuming, ...)
220 │   data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
226 │   batch = next(data_iter)                   # 预取第一个 batch
229 │   wandb.log({"camera_views": [...]}, step=0)  # 可视化图像（健全性检查）
236 │   train_state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
241 │   if resuming: train_state = restore_state(...)
243 │   ptrain_step = jax.jit(
244 │       train_step,
245 │       in_shardings=(replicated, state_sharding, data_sharding),
246 │       out_shardings=(state_sharding, replicated),
247 │       donate_argnums=(1,)                   # 复用 train_state 内存
248 │   )
259 │   for step in tqdm:
260 │       with sharding.set_mesh(mesh):         # 让 activation_sharding_constraint 生效
261 │           train_state, info = ptrain_step(train_rng, train_state, batch)
263 │       if step % log_interval == 0: wandb.log(info)
270 │       batch = next(data_iter)
272 │       if step % save_interval == 0:
273 │           _checkpoints.save_state(ckpt_manager, train_state, data_loader, step)
275 │   ckpt_manager.wait_until_finished()
```

---

## 4. 数据加载层级

```
LeRobotDataset (HF, .arrow shards)
        │
        ▼ __getitem__(idx)
{state, images, actions, prompt(optional), task(optional)}
        │
        ▼ TransformedDataset (transform_dataset, data_loader.py:172)
   ┌────────────────────────────────────────────────────────┐
   │  data_config.repack_transforms.inputs                 │  ← 字段重命名
   │  data_config.data_transforms.inputs                   │  ← AlohaInputs/DroidInputs 等
   │  Normalize(norm_stats, use_quantiles=...)             │  ← state/actions 归一化
   │  data_config.model_transforms.inputs                  │  ← ResizeImages, TokenizePrompt 等
   └────────────────────────────────────────────────────────┘
        │
        ▼ TorchDataLoader（torch DataLoader，多进程）
                num_workers × Dataset[i] → collate → batch
        │
        ▼ jax.tree.map(lambda x: jnp.asarray(x), batch)  + 按 data_sharding 分发
batch: (Observation, Actions)
```

**注意**：数据 transform 在 **torch 多进程 worker** 里跑（不在 JAX 主进程），所以归一化、tokenize、resize 都是 CPU 并行做的，JAX 主进程只做训练前向后向。

详见 [`06-data-transforms.md`](06-data-transforms.md)。

---

## 5. 权重初始化与加载（`init_train_state`）

`train.py:84-133`：

```
init_train_state(config, init_rng, mesh, resume)
        │
        ▼
1. 创建 optimizer (optax.adamw + clip_by_global_norm)
2. 定义 init(rng, partial_params):
       model = config.model.create(rng)        # 实例化 Pi0(...)
       if partial_params:                       # 加载预训练权重的子集
           graphdef, state = nnx.split(model)
           state.replace_by_pure_dict(partial_params)
           model = nnx.merge(graphdef, state)
       params = nnx.state(model)
       # 把冻结参数 cast 到 bf16（节省显存，因为它们不需要梯度）
       params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.astype(bf16))
       return TrainState(step=0, params, model_def, tx, opt_state=tx.init(filtered_params), ema_*)
3. shape_only = jax.eval_shape(init, rng)       # 不真正分配显存，只算 shape
4. state_sharding = sharding.fsdp_sharding(shape_only, mesh, log=True)   # 决定每个 array 怎么切
5. partial_params = config.weight_loader.load(shape_only.params.to_pure_dict())  # 真加载权重
       — CheckpointWeightLoader: 从 gs:// 下载 ckpt
       — PaliGemmaWeightLoader: 加载 PaliGemma 官方 npz
       — NoOpWeightLoader: 啥也不加载（从头训）
6. train_state = jax.jit(init, in_shardings=replicated, out_shardings=state_sharding)(rng, partial_params)
       # 真正在 GPU 上分配显存 + 应用 FSDP 切片
return train_state, state_sharding
```

**核心技巧**：
- 用 `jax.eval_shape` 先算出参数 shape → 决定 FSDP sharding → 再真正初始化。这样**永远不会单卡 OOM**。
- 用 `donate_argnums=(1,)` 让 jit 释放旧的 partial_params buffer。

---

## 6. FSDP 切片策略

`src/openpi/training/sharding.py`：

```
make_mesh(num_fsdp_devices):
   mesh_shape = (device_count // num_fsdp_devices, num_fsdp_devices)
   axes = ("batch", "fsdp")
   return jax.make_mesh(mesh_shape, axes)
```

例：8 卡 + `fsdp_devices=8` → `mesh_shape=(1, 8)`，所有 8 卡都参与 FSDP。
8 卡 + `fsdp_devices=4` → `mesh_shape=(2, 4)`，2 路 DP × 4 路 FSDP。

`fsdp_sharding(pytree, mesh)`（`sharding.py:48`）：
1. 如果 `fsdp_devices=1` → 全 replicate
2. 标量 / 向量 → replicate
3. 小于 `min_size_mbytes=4 MiB` 的 array → replicate
4. 其他：按最大轴切 FSDP（要能整除）；找不到能整除的轴 → replicate

**激活值**呢？用 `activation_sharding_constraint`（`sharding.py:40` 和 `gemma.py` 里多处）：
```python
xs = sharding.activation_sharding_constraint(xs)  # 把激活 along DATA_AXIS 切
```
JIT 编译时 XLA 会自动插入 `all-gather` / `reduce-scatter` 实现 FSDP forward/backward。

---

## 7. train_step 内部

`scripts/train.py:137-191`：

```
train_step(config, rng, state, batch):
    1. model = nnx.merge(state.model_def, state.params)   # 重建模型实例
       model.train()                                       # 设置 deterministic=False（开启 dropout）

    2. loss_fn(model, rng, obs, actions):
           chunked_loss = model.compute_loss(rng, obs, actions, train=True)  # 内部跑 flow matching 损失
           return jnp.mean(chunked_loss)

    3. diff_state = nnx.DiffState(0, config.trainable_filter)  # 只对可训参数求导
       loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng, obs, actions)

    4. params = state.params.filter(config.trainable_filter)
       updates, new_opt_state = state.tx.update(grads, state.opt_state, params)  # AdamW 更新
       new_params = optax.apply_updates(params, updates)

    5. nnx.update(model, new_params)
       new_params = nnx.state(model)          # 把冻结部分合并回来

    6. new_state = TrainState(step+1, new_params, ..., new_opt_state)

    7. if ema_decay:
           new_state.ema_params = α·old_ema + (1-α)·new_params

    8. info = {loss, grad_norm, param_norm}
       return new_state, info
```

**冻结策略**（`freeze_filter`）：
- 全量微调：`freeze_filter = nnx.Nothing` （什么都不冻）
- LoRA 微调：`freeze_filter = nnx.All(PathRegex(".*llm.*"), nnx.Not(PathRegex(".*lora.*")))`
  - 含义：所有 LLM 内部的参数，**除了** LoRA 适配器 → 全部冻结
  - 见 `pi0_fast.py:127 get_freeze_filter`

---

## 8. 优化器（AdamW + clip + 学习率调度）

`src/openpi/training/optimizer.py`：

```python
# 默认 AdamW
AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0)
# 注：weight_decay 设到 1e-10（接近 0）是因为完全设 0 反而 OOM（optax 实现细节）

# 学习率 schedule
CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6)
# 或者
RsqrtDecaySchedule(warmup_steps=1000, peak_lr=5e-5, timescale=10_000)

# create_optimizer 把两者组合：
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # 全局梯度裁剪
    optax.adamw(lr_schedule, ...)
)
```

LoRA 时建议把 `peak_lr` 调到 `1e-4` 量级；全量微调保持 `2.5e-5`。

---

## 9. Orbax 检查点

`src/openpi/training/checkpoints.py`：

```
ckpt_dir/
├── 1000/
│   ├── assets/
│   │   └── <asset_id>/
│   │       └── norm_stats.json   # 跟随 ckpt 保存归一化统计 → 推理时也能用
│   ├── train_state/              # 含 opt_state、step、ema_params（用于恢复训练）
│   └── params/
│       └── params/               # 用于推理的 params（不含 optimizer state）
├── 2000/
├── ...
```

**关键设计**：
- `train_state` 和 `params` 分开存 → 部署只需 `params/`，体积小很多
- `assets/` 用 `CallbackHandler` 写 norm_stats，让 ckpt **自包含归一化统计**
- `max_to_keep=1`：默认只留最新一个（节省存储）
- `keep_period=5000`：每 5000 步永久保留一个

**异步保存**：`CheckpointManager` 后台 thread 异步写盘，不阻塞训练；`wait_until_finished()` 在结束时同步。

---

## 10. 多卡 / 多节点

- 多卡：自然支持。`fsdp_devices=8` 在 8×A100 上跑全量微调 π₀。
- 多节点：**目前不支持**（README 明示）。需要 `jax.distributed.initialize()` + multi-host 但官方没测试过。
- TPU：JAX 路径理论可直接迁，但 Orbax/sharding 需要 tweak。

---

## 11. 一个完整的训练命令例子

```bash
# 1. 计算归一化统计
uv run scripts/compute_norm_stats.py pi0_libero --max-frames 200000

# 2. 启动训练（4 卡 FSDP）
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0_libero \
  --exp-name=my_libero_run \
  --batch-size 64 \
  --num-train-steps 30000 \
  --fsdp-devices 4 \
  --weight-loader.params-path "gs://openpi-assets/checkpoints/pi0_base/params"

# 3. 训练中：tail -f checkpoints/.../log，或 wandb 看板
# 4. 推理：见 05-inference-and-serving.md
```

---

## 12. 训练速度 / 显存经验数据

| 配置                            | GPU      | batch | 显存 / 卡 | step/s | 30k 步耗时 |
| ------------------------------- | -------- | ----- | --------- | ------ | ---------- |
| pi0 LoRA + LIBERO               | RTX 4090 | 8     | 22 GB     | ~2.0   | ~4 h       |
| pi0 full + LIBERO               | A100 80G | 64    | 70 GB     | ~1.5   | ~5.5 h     |
| pi0_fast full + DROID           | 8×H100   | 256   | 60 GB     | ~3.0   | ~3 h       |
| pi05 KI 训练（base 模型）       | TPU pod  | 1024  | -         | -      | 数天       |

（数字均为社区实测，仅供参考）

---

下一篇：[`05-inference-and-serving.md`](05-inference-and-serving.md) — 把训完的 ckpt 跑成 50Hz 机器人 policy。
