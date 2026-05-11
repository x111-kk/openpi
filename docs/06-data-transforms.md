# 06 · 数据 Transforms 三层管线

> openpi 数据流的"中间件层"。从原始机器人数据 → 模型输入 / 模型输出 → 机器人执行动作。理解了这一层就能给**任意自己的机器人**接 π。

---

## 1. 数据流总览

```
LeRobot dataset (HF .arrow) ── episode dict
        │
        ▼   ① repack_transforms.inputs       ← 字段重排（任意 key 重命名/嵌套展开）
        ▼   ② data_transforms.inputs         ← 机器人特定空间转换 (AlohaInputs/DroidInputs/LiberoInputs)
        ▼   ③ Normalize(norm_stats)          ← 用 mean/std 或 q01/q99 归一化 state, actions
        ▼   ④ model_transforms.inputs        ← ResizeImages + TokenizePrompt(...)
batch (Observation, Actions)
        │
        ▼   train / sample_actions
model outputs
        │
        ▼   ④' model_transforms.outputs      ← (通常空，FAST 模型这里做 detokenize)
        ▼   ③' Unnormalize(norm_stats)       ← 反归一化 actions
        ▼   ②' data_transforms.outputs       ← AlohaOutputs/DroidOutputs (joint flip 反向、gripper 反向)
        ▼   ①' repack_transforms.outputs     ← 字段重命名 (通常空)
returned to robot
```

**对应代码**：
- 串联：`src/openpi/training/data_loader.py:172` 的 `transform_dataset`
- 推理：`src/openpi/policies/policy_config.py:create_trained_policy`
- 基类：`src/openpi/transforms.py:23` 的 `DataTransformFn` Protocol

---

## 2. 第①层：`repack_transforms` —— 字段重排

最简单的一个 transform：把任意嵌套 dict 重组成 openpi 期望的扁平结构。

`transforms.py:80`：
```python
@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    structure: PyTree[str]
    def __call__(self, data):
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)
```

例子（DROID）：把 LeRobot 的 `observation.images.exterior_image_1_left` 改名为 `observation/exterior_image_1_left`：
```python
RepackTransform({
    "observation/exterior_image_1_left": "observation.images.exterior_image_1_left",
    "observation/wrist_image_left":      "observation.images.wrist_image_left",
    "observation/joint_position":        "observation.state.joint_position",
    "observation/gripper_position":      "observation.state.gripper_position",
    "actions":                            "action",
    "prompt":                             "task",
})
```

接 **自己的机器人** 时，先用这个 transform 把 dict 调成 openpi 风格，避免改后续 transform。

---

## 3. 第②层：`data_transforms` —— 机器人特定空间转换

**职责**：把机器人原生的关节角 / 末端坐标 / gripper 转成 π 内部的统一 32-DoF 动作空间，反过来也一样。

### 3.1 AlohaInputs（双臂 ALOHA, 14-DoF）

`src/openpi/policies/aloha_policy.py:25`：

```python
@dataclasses.dataclass(frozen=True)
class AlohaInputs(DataTransformFn):
    adapt_to_pi: bool = True       # 是否做坐标转换（用 π 预训练的 ALOHA 时设 True）
    EXPECTED_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data):
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
        # 1. 摄像头映射
        images = {
            "base_0_rgb":         data["images"]["cam_high"],   # 顶视
            "left_wrist_0_rgb":   data["images"].get("cam_left_wrist",  zeros),
            "right_wrist_0_rgb":  data["images"].get("cam_right_wrist", zeros),
        }
        image_masks = {名字: 该摄像头是否真实存在}
        # 2. state pass-through (14 维)
        inputs = {"image": images, "image_mask": image_masks, "state": data["state"]}
        if "actions" in data:
            inputs["actions"] = _encode_actions_inv(data["actions"], adapt_to_pi)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs
```

**`adapt_to_pi` 在干嘛？**
1. **关节方向翻转**：`_joint_flip_mask() = [1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1]`
   - 因为 π 的 ALOHA 与原版 ALOHA 关节正方向定义不同，乘以这个 mask 翻转。
2. **Gripper 角度转换**：`_gripper_to_angular(linear_pos)`
   - ALOHA 用线性位置（米），π 用 gripper 关节角（弧度）；做线性 → 弧度的三角换算。
   - 数学：`linear_to_radian(l, a, h) = arcsin((h² + l² - a²) / (2hl))`（柄长 h、连杆 a）。

**反方向**（输出，`AlohaOutputs`）：`_encode_actions` 把 π 预测的 14-DoF 动作翻回 ALOHA 坐标系。

### 3.2 DroidInputs（单臂 Franka, 8-DoF）

`src/openpi/policies/droid_policy.py:31`：

```python
class DroidInputs(DataTransformFn):
    model_type: ModelType   # π0 / π0.5 / π0-FAST

    def __call__(self, data):
        # 1. 拼接 7-DoF joint + 1-DoF gripper = 8-DoF state
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])

        base_image  = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        # 2. 不同模型用不同摄像头 slot
        if model_type in (PI0, PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, zeros)
            masks  = (True, True, False)   # 右手腕摄像头不存在 → mask False
        elif model_type == PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, zeros, wrist_image)
            masks  = (True, True, True)    # FAST 不 mask（语义吃 padding）

        return {"state": state, "image": ..., "image_mask": ..., "prompt": ...}
```

**输出**（`DroidOutputs`）：只取动作的前 8 维（π 输出 32 维统一空间，DROID 只用前 8）：
```python
return {"actions": data["actions"][:, :8]}
```

### 3.3 LiberoInputs（单臂 Franka, 7-DoF）

`src/openpi/policies/libero_policy.py:30`：基本是 DROID 的简化版，state=8 维（含 gripper），actions 输出取前 7（含 gripper 二元）。

注释非常友好（"自己写时复制这个类，按注释改 keys"），适合作为**新机器人 policy 的模板**。

### 3.4 自己写一个 policy 类

```python
# my_robot_policy.py
import dataclasses
import numpy as np
import einops
from openpi import transforms
from openpi.models import model as _model

@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data):
        # state: pad/truncate to model action_dim if 必要
        state = np.asarray(data["state"], dtype=np.float32)

        # images: (H, W, 3) uint8
        base = self._parse_img(data["images"]["base"])
        wrist = self._parse_img(data["images"]["wrist"])

        return {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            **({"actions": data["actions"]} if "actions" in data else {}),
            **({"prompt": data["prompt"]} if "prompt" in data else {}),
        }

    @staticmethod
    def _parse_img(x):
        x = np.asarray(x)
        if x.dtype.kind == "f": x = (255*x).astype(np.uint8)
        if x.shape[0] == 3:    x = einops.rearrange(x, "c h w -> h w c")
        return x

@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"][:, :MY_ACTION_DIM])}
```

然后在 `training/config.py` 里加个新的 `DataConfigFactory`：
```python
class MyRobotDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(inputs=[transforms.RepackTransform({...})])
        data = transforms.Group(
            inputs=[MyRobotInputs(model_type=model_config.model_type)],
            outputs=[MyRobotOutputs()],
        )
        return self.create_base_config(assets_dirs).replace(
            repack_transforms=repack,
            data_transforms=data,
        )
```

加完就能直接 `uv run scripts/train.py my_robot_config` 训练，详见 [`10-finetune-cookbook.md`](10-finetune-cookbook.md)。

---

## 4. 第③层：`Normalize` —— 归一化

`transforms.py:115`：

```python
class Normalize(DataTransformFn):
    norm_stats: PyTree[NormStats]
    use_quantiles: bool = False

    def _normalize(self, x, stats):
        # 标准 z-score
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats):
        # 分位数归一化（鲁棒到 [-1, 1]）
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
```

`NormStats` 来自 `compute_norm_stats.py` 跑出的 JSON（详见 `04-training-pipeline.md` §1）。

**为什么用 q01/q99 而不是 mean/std？**
- 机器人数据常有离群值（碰撞失败 demo、操作员手抖）；q01/q99 比 mean/std 鲁棒。
- 归一化到 `[-1, 1]` 后离群值会被截断到 ±1（避免梯度爆炸）。

`Unnormalize` 是反向操作，注意它对维度的扩展处理：如果 norm_stats 维度小于实际，余下维度按"原样不归一化"返回（用于 actions 比 stats 维度大的情况）。

---

## 5. 第④层：`model_transforms` —— 模型输入准备

### 5.1 `ResizeImages`

`transforms.py:185`：调用 `image_tools.resize_with_pad`（保留长宽比，pad 黑边）。所有图像 → `(224, 224)`。

### 5.2 `TokenizePrompt`

`transforms.py:248`：
```python
class TokenizePrompt(DataTransformFn):
    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = False  # π0.5 设 True，把 state 也编进文本

    def __call__(self, data):
        prompt = data.pop("prompt")
        state = data.get("state") if self.discrete_state_input else None
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        data["tokenized_prompt"] = tokens
        data["tokenized_prompt_mask"] = mask
        return data
```

### 5.3 `TokenizeFASTInputs`（仅 π₀-FAST）

把 state + actions 全部编码进 token 序列（含 FAST 离散化），多输出 `token_ar_mask`、`token_loss_mask`。详见 [`09-tokenizers.md`](09-tokenizers.md)。

### 5.4 `InjectDefaultPrompt`

`transforms.py:105`：obs 没带 prompt 时注入默认值。常用在远程 server 端（机器人可能忘传 prompt）。

---

## 6. 一组完整的 transform 配置示例

`training/config.py` 的 `LeRobotAlohaDataConfig`（行号 229+）：

```python
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True

    @override
    def create(self, assets_dirs, model_config):
        # ① repack
        repack = Group(inputs=[
            RepackTransform({
                "images":  {"cam_high": "observation.images.top", ...},
                "state":   "observation.state",
                "actions": "action",
            })
        ])
        # ② data transforms
        data = Group(
            inputs=[AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        # 可选 delta action 转换
        if self.use_delta_joint_actions:
            delta_mask = make_bool_mask(6, -1, 6, -1)   # 翻 joint dims，但 gripper 保留绝对
            data = data.push(
                inputs=[DeltaActions(mask=delta_mask)],
                outputs=[AbsoluteActions(mask=delta_mask)],
            )
        # ③ model transforms
        model_tf = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        # ④ 组装
        return self.create_base_config(assets_dirs).replace(
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=model_tf,
        )
```

`ModelTransformFactory`（`config.py:107`）：根据模型类型自动加 `ResizeImages` + 合适的 tokenizer（`PaligemmaTokenizer` for π0/π05，`FASTTokenizer` for π0-FAST）。

---

## 7. Delta vs Absolute Actions

`transforms.py:204` `DeltaActions`：
```python
actions[..., :dims] -= state[..., :dims]   # 转 delta（相对当前 state）
```
`transforms.py:226` `AbsoluteActions`：反向。

**为什么要 delta？**
- 当 state 范围很大（如机器人在很大工作空间运动），绝对 actions 难学。
- Delta 让模型只学增量 → 数值范围紧凑、泛化更好。
- Gripper 通常仍用绝对（开 / 关是离散的）。

`make_bool_mask(6, -1, 6, -1) = [T, T, T, T, T, T, F, T, T, T, T, T, T, F]` —— 前 6 个 joint delta、第 7 个 gripper 绝对、再 6 个 joint delta、最后 gripper 绝对（ALOHA 双臂结构）。

---

## 8. transform 设计原则（自己写时遵循）

1. **纯函数 / dataclass(frozen=True)**：所有 transform 都是不可变 dataclass，便于序列化、缓存、分布式分发。
2. **可组合**：用 `Group.push(inputs=[...], outputs=[...])` 在已有 transform 上加层，**output 是反向 prepend**（保证 input 顺序的逆序应用）。
3. **leaf-by-leaf**：底层用 `jax.tree.map` 操作 leaf array，结构不变。
4. **不在 transform 里做模型相关计算**：transform 只准备数据，模型 forward 在 `model.compute_loss / sample_actions` 里。

---

## 9. 调试 transform 的小技巧

```python
# 在 Python REPL 里手动跑一遍
from openpi.training.config import get_config
config = get_config("pi0_libero")
data_config = config.data.create(config.assets_dirs, config.model)

from openpi.policies.libero_policy import make_libero_example
ex = make_libero_example()

# 走 transform pipeline
from openpi import transforms
input_tf = transforms.compose([
    *data_config.repack_transforms.inputs,
    *data_config.data_transforms.inputs,
    transforms.Normalize(data_config.norm_stats),
    *data_config.model_transforms.inputs,
])
out = input_tf(ex)
print({k: v.shape if hasattr(v,'shape') else v for k,v in out.items()})
```

输出应包含 `image`, `image_mask`, `state`, `actions`, `tokenized_prompt`, `tokenized_prompt_mask`。

---

下一篇：[`07-key-techniques.md`](07-key-techniques.md) — LoRA / AdaRMS / KV cache / gemma_fast 源码精读。
