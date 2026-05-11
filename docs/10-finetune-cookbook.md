# 10 · 从零 fine-tune 自己的机器人 · Cookbook

> 给你自己机器人的数据 + π₀ 预训练 checkpoint → 一个真机能跑的 policy。
>
> 含具体命令、配置代码、显存估算、踩坑提示。**这一篇可以当一个"操作手册"对着抄。**

---

## 0. 前置条件

- Ubuntu 22.04，CUDA 12，NVIDIA driver ≥ 535
- GPU：4090（22GB+）做 LoRA，A100/H100 做 full
- Python 3.11（不是 3.12！tf-cpu 2.15 还没出 cp312 wheel）
- 200GB+ 磁盘（base ckpt ~12GB + 数据集 + 训练 ckpts）
- 一份 LeRobot 格式 / RLDS 格式 的真机演示数据（建议 ≥ 50 episodes, ≥ 5 min/episode）

---

## 1. 安装

```bash
git clone --recurse-submodules https://github.com/x111-kk/openpi.git
cd openpi/references/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
# 装训练用的可选组（如果用 DROID RLDS 数据需要）
uv sync --group rlds
```

验证：
```bash
uv run python -c "from openpi.models.pi0_config import Pi0Config; print(Pi0Config())"
```

---

## 2. 准备数据集（LeRobot 格式）

LeRobot 格式约定：
```
my_robot_data/
├── meta/
│   ├── info.json        # 描述字段名、shape、dtype
│   ├── stats.json       # 每列 mean/std/min/max（LeRobot 自带，不是 openpi 用）
│   ├── tasks.jsonl      # 任务名列表
│   └── episodes.jsonl   # episode 索引
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

如果你的数据不是 LeRobot 格式，最快方法：
```bash
# 用 LeRobot 提供的转换工具
pip install lerobot
lerobot-convert ... # 看 LeRobot 文档
```

或者写一个 PyTorch `Dataset` 实现 `__len__` + `__getitem__` 返回 dict，挂到 openpi 自己的 `IterableDataset` 协议。

最小数据要求（推荐）：
- ≥ 50 episodes
- ≥ 5000 步 × action_horizon=50 = 25 万样本（fine-tune）
- ≥ 1 个第三人称摄像头 + 1 个 wrist 摄像头
- proprio state（关节角 + gripper）
- 每条 episode 一个语言任务描述（即使简单）

把数据放到 HF Hub 或本地 `~/.cache/huggingface/lerobot/<repo_id>/`。

---

## 3. 写自己的 DataConfig

在 `references/openpi/src/openpi/training/config.py` 末尾的 `_CONFIGS = [...]` 之前 / 自己 fork 里加：

### 3.1 Step 1：定义 transform

新建文件 `references/openpi/src/openpi/policies/myrobot_policy.py`：

```python
import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_myrobot_example() -> dict:
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data):
        base = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # FAST 不 mask；flow matching 模型 mask 掉 padding 图
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        # 假设你的机器人 7-DoF（6-DoF joint + gripper）
        return {"actions": np.asarray(data["actions"][:, :7])}
```

### 3.2 Step 2：定义 DataConfigFactory

在 `training/config.py` 里加：

```python
class LeRobotMyRobotDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs, model_config):
        from openpi.policies import myrobot_policy

        repack = transforms.Group(inputs=[
            transforms.RepackTransform({
                "observation/state":         "observation.state",
                "observation/image":         "observation.images.top",
                "observation/wrist_image":   "observation.images.wrist",
                "actions":                    "action",
                "prompt":                     "task",
            })
        ])

        data = transforms.Group(
            inputs=[myrobot_policy.MyRobotInputs(model_type=model_config.model_type)],
            outputs=[myrobot_policy.MyRobotOutputs()],
        )

        # 可选：actions 用 delta（保留 gripper 绝对）
        delta_mask = transforms.make_bool_mask(6, -1)   # 前 6 joint delta, 第 7 gripper absolute
        data = data.push(
            inputs=[transforms.DeltaActions(mask=delta_mask)],
            outputs=[transforms.AbsoluteActions(mask=delta_mask)],
        )

        return self.create_base_config(assets_dirs).replace(
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=ModelTransformFactory()(model_config),
            use_quantile_norm=True,         # 推荐：分位数归一化
        )
```

### 3.3 Step 3：加 TrainConfig

```python
TrainConfig(
    name="pi0_myrobot_lora",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",      # ← LoRA 主干
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
    ),
    data=LeRobotMyRobotDataConfig(
        repo_id="x111-kk/my-robot-fold-towel",   # 你的 HF 数据集 id（或本地路径）
        asset_id="my_robot",                      # norm_stats 保存子目录名
        base_config=DataConfig(
            local_files_only=False,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    optimizer=_optimizer.AdamW(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=1e-4,                              # LoRA 用大点的 lr
        decay_steps=10_000,
        decay_lr=1e-5,
    ),
    freeze_filter=Pi0Config(...).get_freeze_filter(),  # 自动算 LoRA freeze
    num_train_steps=10_000,
    batch_size=32,
    fsdp_devices=1,
    project_name="my-robot",
),
```

---

## 4. 计算归一化统计

```bash
cd references/openpi
uv run scripts/compute_norm_stats.py pi0_myrobot_lora --max-frames 100000
```

会生成 `assets/<asset_id>/norm_stats.json`。检查里面 `mean / std / q01 / q99` 是否合理。

---

## 5. 启动训练

### 5.1 单卡 LoRA（4090）

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_TRACEBACK_FILTERING=off \
uv run scripts/train.py pi0_myrobot_lora \
  --exp-name=fold_towel_v1 \
  --batch-size=8 \
  --num-train-steps=10000 \
  --fsdp-devices=1
```

实测显存 ~22 GB，~2.0 step/s，10k 步约 1.5 小时。

### 5.2 多卡 full fine-tune（4×A100）

把 config 改成 `paligemma_variant="gemma_2b"`（不带 lora）+ `freeze_filter=nnx.Nothing`：

```bash
uv run scripts/train.py pi0_myrobot_full \
  --exp-name=fold_towel_full \
  --batch-size=64 \
  --fsdp-devices=4 \
  --num-train-steps=30000
```

显存 ~70 GB / 卡，30k 步约 5 小时。

### 5.3 W&B 监控

训练自动打开 wandb（前提是 `wandb login` 过）。看：
- `loss` 应 < 0.1（一般 ~0.02-0.05）
- `grad_norm` < 1（如果 > 10 说明数据/lr 有问题）
- `param_norm` 平稳，不爆炸

---

## 6. 推理验证

### 6.1 离线（直接读 ckpt）

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.policies.myrobot_policy import make_myrobot_example

config = get_config("pi0_myrobot_lora")
policy = create_trained_policy(
    config,
    checkpoint_dir="checkpoints/pi0_myrobot_lora/fold_towel_v1/9000",
)

obs = make_myrobot_example()
obs["prompt"] = "fold the towel"
out = policy.infer(obs)
print(out["actions"].shape)  # (50, 7) -- 因为 MyRobotOutputs 截断到 7
```

### 6.2 开 WebSocket server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_myrobot_lora \
  --policy.dir=checkpoints/pi0_myrobot_lora/fold_towel_v1/9000 \
  --port=8000 \
  --default-prompt="fold the towel"
```

### 6.3 机器人端 client

参考 [`05-inference-and-serving.md`](05-inference-and-serving.md) §5 的代码模板。

---

## 7. 常见问题排查

### 7.1 训练 loss 不下降

| 现象                          | 原因                                                  | 解决                                                       |
| ----------------------------- | ----------------------------------------------------- | ---------------------------------------------------------- |
| loss 卡在 1.0 附近            | norm_stats 没生效                                     | 检查 `assets/<asset_id>/norm_stats.json` 是否存在          |
| loss 周期性跳动               | batch 太小 / 数据多样性差                             | batch ≥ 32；增加数据多样性                                 |
| loss 第一步就 NaN             | dtype 问题 / 学习率太大                               | 用 bf16；lr ≤ 5e-5 全量、1e-4 LoRA                         |
| grad_norm 爆炸                | gripper / 离群值                                      | 用 `use_quantile_norm=True`                                |
| Wandb 显示图像全黑            | _parse_image 没正确处理 channel/dtype                 | 在 transform 输出后 plt.imshow 验证                        |

### 7.2 推理表现差（真机）

| 现象                          | 原因                                                  | 解决                                                       |
| ----------------------------- | ----------------------------------------------------- | ---------------------------------------------------------- |
| 动作完全乱                    | 训练时的 transform vs 推理时不一致                    | 推理用同一个 `pi0_myrobot_lora` config（推荐 `create_trained_policy`） |
| 动作幅度很小（"颤抖"）        | norm_stats 反归一化错误                               | 检查 `Unnormalize` 用的 q01/q99 与训练一致                 |
| 动作有规律但慢一拍            | action_horizon 太长 / 没用 RTC                        | 改用 `action_chunk_broker` 重叠规划                        |
| 任务执行到一半失败            | 训练数据缺这种 corner case                            | 增加 demonstrations，特别是失败恢复轨迹                    |
| 完全无视语言指令              | 单任务模型 / prompt tokenize 错                       | 多任务训练；检查 `tokenized_prompt` 内容                  |

### 7.3 显存爆

| 现象                          | 解决                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| 单 4090 装不下 LoRA           | batch_size=4；torch.cuda.empty_cache()；XLA_PYTHON_CLIENT_PREALLOCATE=false                            |
| FSDP 多卡仍然 OOM             | `fsdp_devices=8`（用满所有卡）；remat_policy="full"                                                    |
| OOM 但 nvidia-smi 看着不满    | XLA 预分配 90% 显存。设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`   |

---

## 8. 推荐的训练超参（fine-tune）

| 数据规模           | 学习率（LoRA） | 学习率（full） | warmup | steps   | batch | EMA  |
| ------------------ | -------------- | -------------- | ------ | ------- | ----- | ---- |
| 50 episodes        | 1e-4           | 2.5e-5         | 500    | 5,000   | 16    | 0.99 |
| 200 episodes       | 1e-4           | 2.5e-5         | 1,000  | 10,000  | 32    | 0.99 |
| 1000+ episodes     | 5e-5           | 1e-5           | 2,000  | 30,000  | 64    | 0.99 |

---

## 9. 整套 fine-tune 时间表

| 阶段                       | 工作                                                   | 时间预算    |
| -------------------------- | ------------------------------------------------------ | ----------- |
| 0. 数据采集                | 真机演示 50-200 episodes                                | 1-3 天      |
| 1. 数据转 LeRobot 格式     | 写脚本转 LeRobot；上传到 HF Hub                         | 半天        |
| 2. 写 policy + DataConfig  | `myrobot_policy.py` + `training/config.py`              | 半天        |
| 3. 计算 norm stats         | `compute_norm_stats.py`                                 | 10 分钟     |
| 4. 跑 debug 训练           | `--num-train-steps=100` 检查 forward 正常              | 10 分钟     |
| 5. 正式 fine-tune（LoRA）  | 1×4090 跑 10k 步                                       | 2 小时      |
| 6. 离线推理验证            | jupyter notebook 跑 `make_*_example` + `policy.infer`  | 1 小时      |
| 7. 真机部署 + WebSocket    | 启动 server，机器人 client 连接，跑 1 个 task          | 1 小时      |
| 8. 迭代（看真机效果调参）  | 加更多 demos / 改 prompt / 调 lr                       | 数天        |

**最快从零到真机跑通：3-5 天**（数据已有的情况下）。

---

## 10. 进阶：多机器人多任务联合训练

把多个 `DataConfigFactory` 合并：

```python
class MultiRobotDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs, model_config):
        cfg1 = LeRobotAlohaDataConfig(...).create(...)
        cfg2 = LeRobotMyRobotDataConfig(...).create(...)
        # 用 LeRobot 的 mixed dataset 接口拼起来 ...
```

参考 `roboarena_config.py` 看 P&I 如何做多数据集采样权重。

---

## 11. Cheat Sheet

```bash
# 1. 装
git clone --recurse-submodules https://github.com/x111-kk/openpi.git
cd openpi/references/openpi && uv sync

# 2. 准备数据 + 写 config（按本篇 §2-3）

# 3. 算 norm
uv run scripts/compute_norm_stats.py pi0_myrobot_lora

# 4. 训
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0_myrobot_lora \
  --exp-name=v1 --batch-size=8 --num-train-steps=10000 \
  --weight-loader.params-path "gs://openpi-assets/checkpoints/pi0_base/params"

# 5. 推理
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_myrobot_lora \
  --policy.dir=checkpoints/pi0_myrobot_lora/v1/9000 \
  --port=8000

# 6. 机器人端
pip install openpi-client
python my_robot_loop.py  # 参考 docs/05 §10
```

---

## 12. 学习资源 / 当遇到问题

- 官方仓库 README + `docs/` 子目录：<https://github.com/Physical-Intelligence/openpi>
- 官方 Discord：<https://discord.gg/physicalintelligence>
- 论文（必读）：
  - π₀：<https://www.physicalintelligence.company/blog/pi0>
  - π₀-FAST：<https://www.physicalintelligence.company/research/fast>
  - π₀.₅：<https://www.physicalintelligence.company/blog/pi05>
  - Knowledge Insulation：<https://www.physicalintelligence.company/research/knowledge_insulation>

完结。从这里开始动手实验吧。

---

> 全部 10 篇 deep-dive 完成。配合 `references/openpi/` 子模块的源码 + 你的 IDE，应该能从概念到实操全打通。
