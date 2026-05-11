# 08 · 版本演进与源码 diff（π₀ → π₀-FAST → π₀.₅ → π₀.₆）

> 把 Physical Intelligence 公开的 4 个 π 模型在**源码层面**逐项对比。理解了 diff 就理解了每次迭代的工程贡献。

---

## 1. 版本时间线（含闭源 π₀.₆）

```
2024-10   π₀         首发：flow matching + dual-tower MoE + PaliGemma 3B + 50-step action chunk
2024-12   π₀-FAST    FAST 离散动作 tokenizer → 纯自回归训练，可秒收敛但推理慢
2025-02   Hi Robot   "VLA + 高层 LLM"双系统设计（非新模型，是 stack）
2025-04   π₀.₅       AdaRMS 时间条件 + Knowledge Insulation 训练
2025-05   KI         单独发布的 Knowledge Insulation 训练范式（论文）
2025-06   RTC        Real-Time Chunking 算法（部署时延平滑）
2025-09   PyTorch 支持  openpi 增加 PyTorch 路径
2025-11   π₀.₆ / π*₀.₆ 闭源；据博客提到的进展：更大数据 + 更强 VLM backbone + 通用化
```

---

## 2. 配置矩阵（在 openpi 中可用的）

| 模型             | 模型类         | Config 类             | VLM            | Action expert   | Tokenizer            | 推理方式         | Action 输出       |
| ---------------- | -------------- | --------------------- | -------------- | --------------- | -------------------- | ---------------- | ----------------- |
| π₀               | `Pi0`          | `Pi0Config`           | `gemma_2b`     | `gemma_300m`    | `PaligemmaTokenizer` | Flow matching ODE | 连续, 50×32       |
| π₀ LoRA          | `Pi0`          | `Pi0Config`           | `gemma_2b_lora`| `gemma_300m`    | 同上                 | 同上             | 同上              |
| π₀-FAST          | `Pi0FAST`      | `Pi0FASTConfig`       | `gemma_2b`     | （无）          | `FASTTokenizer`      | 自回归 decode    | 离散 token → 连续 |
| π₀-FAST LoRA     | `Pi0FAST`      | `Pi0FASTConfig`       | `gemma_2b_lora`| （无）          | 同上                 | 同上             | 同上              |
| π₀.₅             | `Pi0` (pi05=True)| `Pi0Config(pi05=True)`| `gemma_2b`   | `gemma_300m`    | `PaligemmaTokenizer` | Flow matching + AdaRMS | 连续, 50×32 |
| π₀.₆             | （未开源）     |                       |                |                 |                      |                  |                   |

---

## 3. π₀ vs π₀.₅ 源码 diff（只有 8 处不同）

### 3.1 模型类

**同一个 `Pi0` 类**（`pi0.py`），通过 `Pi0Config.pi05=True` 切换。`pi0_config.py`:

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    pi05: bool = False                    # ← 关键开关
    paligemma_variant: Variant = "gemma_2b"
    action_expert_variant: Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
```

### 3.2 Gemma Module 启用 AdaRMS

`pi0.py:73`：
```python
llm = nnx_bridge.ToNNX(_gemma.Module(
    configs=[paligemma_config, action_expert_config],
    embed_dtype=config.dtype,
    use_adarms=[False, True] if config.pi05 else None,   # ← 关键差异
))
```

- π₀：`use_adarms=None`，所有 expert 用 vanilla RMSNorm
- π₀.₅：`use_adarms=[False, True]`，只 action expert 开 AdaRMS

### 3.3 Action expert 接收时间条件

`pi0.py:140 embed_suffix`：

```python
def embed_suffix(self, obs, noisy_actions, timestep):
    # ... state token, action tokens ...
    time_emb = posemb_sincos(timestep, ...)

    if self.config.pi05:
        # 时间作为 cond 传给 RMSNorm，不再 concat 到 action token
        action_tokens = self.action_in_proj(noisy_actions)        # 注意：没有拼 time
        cond = self.time_mlp(time_emb)                            # cond shape: [B, 256]
    else:
        # 时间 concat 到 action embedding（π₀ 原始做法）
        action_tokens = jnp.concatenate([time_emb_broadcasted, noisy_actions], axis=-1)
        action_tokens = self.action_in_proj(action_tokens)
        cond = None
    return tokens, mask, ar_mask, cond
```

### 3.4 输入 prompt 包含 state（discrete_state_input）

π₀.₅ 把 proprio state 也编码进 prompt 文本，让 VLM 能"看见"机器人当前关节角：

`transforms.py:248 TokenizePrompt`：
```python
class TokenizePrompt(DataTransformFn):
    discrete_state_input: bool = False    # ← π₀.₅ 时设 True

    def __call__(self, data):
        prompt = data.pop("prompt")
        state = data.get("state") if self.discrete_state_input else None
        tokens, mask = self.tokenizer.tokenize(prompt, state)
```

`tokenizer.py:14 PaligemmaTokenizer.tokenize(prompt, state)`：
```python
def tokenize(self, prompt, state=None):
    if state is not None:
        # state 离散化到 256 bin
        discretized = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, discretized))
        prompt = f"Task: {prompt}, State: {state_str};\n"
    return self._sp.encode(prompt, add_bos=True), [True] * len(tokens)
```

### 3.5 Loss 计算（保持不变）

`pi0.py:189 compute_loss`：

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = preprocess_observation(preprocess_rng, observation, train=train)
    batch_shape = actions.shape[:-2]

    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(observation, x_t, time)
    # ... forward 拿 v_t ...
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

π₀ / π₀.₅ 完全一样。**π₀.₅ 的差异全部在 embed_suffix 和 AdaRMS**。

### 3.6 KI（Knowledge Insulation）训练范式

π₀.₅ 引入的训练 trick（不是模型差异，是 loss 差异）：
- 训练时**对 VLM expert 的梯度做特殊处理**（stop_gradient 或单独 sub-loss），让它的表征不被 action 损失污染。
- openpi 中 KI 由 `freeze_filter` + `data_transforms` 实现：
  - 第一阶段：纯 VLM 任务（图像-文本配对）训 VLM expert
  - 第二阶段：冻结 VLM，只训 action expert + AdaRMS
  - 第三阶段（可选）：极小学习率联合训练
- 论文：<https://www.physicalintelligence.company/research/knowledge_insulation>

---

## 4. π₀ vs π₀-FAST 源码 diff（差异最大）

### 4.1 不同的模型类（`pi0_fast.py`）

`Pi0FAST` 是**纯自回归 transformer**，没有 action expert，没有 flow matching：

```python
@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(BaseModelConfig):
    paligemma_variant: Variant = "gemma_2b"
    action_dim: int = 32
    action_horizon: int = 32                     # 注意：不是 50
    max_token_len: int = 250                     # 含 action token，更长
    fast_model_tokenizer: Any | None = None      # FAST tokenizer
```

### 4.2 输入 / 输出全是 token

`pi0_fast.py:160 embed_inputs`：

```python
def embed_inputs(self, observation):
    # 1. 把图像 patch 编码成 token（同 π₀）
    img_tokens = self.PaliGemma.img(observation.images)
    # 2. 用 FAST tokenizer 把 (prompt + state + actions) 编码成 token 序列
    text_tokens = observation.tokenized_prompt    # 来自 FAST tokenizer
    # 3. 拼接图像和文本 token
    return tokens, mask, ar_mask
```

### 4.3 训练时是标准 next-token loss

`pi0_fast.py:198 compute_loss`：

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # 1. embed
    tokens, mask, ar_mask = self.embed_inputs(observation)
    # 2. forward
    logits, _ = self.gemma_fast.forward(tokens, mask, ar_mask)
    # 3. shifted cross-entropy on action tokens only
    targets = jnp.roll(tokens, shift=-1, axis=1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = loss * observation.token_loss_mask        # 只在 action token 上计 loss
    return loss
```

### 4.4 推理是自回归 decode

`pi0_fast.py:236 sample_actions`：

```python
def sample_actions(self, rng, observation, *, max_decoding_steps=128, temperature=0.0):
    # 1. encode 输入 prefix
    tokens, mask = self.tokenizer.tokenize_inference(observation)
    logits, kv_cache = self.gemma_fast.forward(tokens, kv_cache=None)

    # 2. 自回归循环
    cur_token = jnp.argmax(logits[:, -1, :]) if temperature == 0 else sample(logits[:, -1, :] / temperature)
    out_tokens = [cur_token]

    for step in range(max_decoding_steps):
        logits, kv_cache = self.gemma_fast.forward(cur_token, kv_cache=kv_cache)
        cur_token = ...
        if cur_token == EOS_TOKEN: break
        out_tokens.append(cur_token)

    # 3. detokenize → 连续 actions
    return self.fast_tokenizer.extract_actions(out_tokens, action_horizon, action_dim)
```

### 4.5 性能对比（社区实测）

| 指标             | π₀（flow matching）| π₀-FAST（自回归）     |
| ---------------- | ------------------- | --------------------- |
| 训练收敛速度     | ~30k 步             | ~10k 步（更快）       |
| 推理延迟（H100） | ~180 ms             | ~250 ms（更慢，128 token 解码）|
| Action 精度      | 连续（无量化误差）  | 离散（DCT 编码近似）  |
| 适合场景         | 实时控制            | 离线规划 / fine-tune 容易 |

**结论**：
- 学术研究 / 快速 fine-tune → π₀-FAST（训得快，无需 flow matching 复杂度）
- 生产部署 / 实时机器人 → π₀ / π₀.₅（推理快，精度高）

---

## 5. 关键 Config 字段速查

| 字段                       | π₀         | π₀.₅                  | π₀-FAST    |
| -------------------------- | ---------- | --------------------- | ---------- |
| `pi05`                     | False      | **True**              | -          |
| `paligemma_variant`        | gemma_2b   | gemma_2b              | gemma_2b   |
| `action_expert_variant`    | gemma_300m | gemma_300m            | -          |
| `use_adarms`               | None       | **[False, True]**     | -          |
| `action_dim`               | 32         | 32                    | 32         |
| `action_horizon`           | 50         | 50                    | **32**     |
| `max_token_len`            | 48         | 48                    | **250**    |
| `discrete_state_input`     | False      | **True**              | True (内置)|
| Loss                       | MSE on v_t | MSE on v_t            | CE on tokens|

---

## 6. 各版本的工程贡献小结

### π₀（2024-10）
- 首次把 PaliGemma VLM 与 Action expert 用 MoE 共享 attention 串起来
- Flow matching 应用到 action chunk 生成（不是单 step action）
- 50 步 action chunk → 1 秒控制粒度

### π₀-FAST（2024-12）
- DCT + BPE 把连续动作变成离散 token
- 让纯 LLM 训练范式可以直接用于 VLA → 与 HuggingFace 生态完全兼容
- 论文：<https://www.physicalintelligence.company/research/fast>

### Hi Robot（2025-02）
- 系统级：把 π₀ 作为低层 policy，叠加 LLM 作为高层 planner（"thinker"）
- 不是新模型，是 stack 设计

### π₀.₅（2025-04）
- AdaRMS：时间条件注入到每层 RMSNorm
- KI 训练：分阶段训，保护 VLM 知识
- discrete_state_input：把 state 也编进 prompt，提升泛化

### KI 论文（2025-05）
- 把 KI 抽象成通用训练范式，用于其他 VLA 模型
- 论文：<https://www.physicalintelligence.company/research/knowledge_insulation>

### RTC（2025-06）
- 部署算法：用 chunk 重叠平滑动作过渡
- 不改模型，是推理时的 wrapper
- 论文：<https://www.physicalintelligence.company/research/real_time_chunking>

### PyTorch 支持（2025-09）
- `src/openpi/models_pytorch/` 全套 PyTorch 实现
- 配合 HF transformers + FSDP，降低入门门槛

### π₀.₆ / π*₀.₆（2025-11，闭源）
- 公开博客提到：更大规模数据（含网页 + 视频）、更强 VLM backbone（可能 Gemma 2 / Llama-Vision）、通用零样本任务
- 未在 openpi 开源；社区参考的最新版仍是 π₀.₅

---

## 7. 如何选模型？

```
任务定义
  │
  ├─ 实时机器人（50Hz 控制） ────→ π₀ / π₀.₅
  │                                  │
  │                                  ├─ 显存 < 30 GB → π₀ LoRA / π₀.₅ LoRA
  │                                  └─ 显存 ≥ 70 GB → π₀ full / π₀.₅ full
  │
  ├─ 离线规划 / 长程任务 ─────→ Hi Robot stack（π₀ + LLM planner）
  │
  ├─ 快速 fine-tune 学术研究 ──→ π₀-FAST（训练快，HF 生态友好）
  │
  └─ 多语言 / 多任务零样本 ───→ π₀.₅（discrete_state + KI 训练）
```

---

下一篇：[`09-tokenizers.md`](09-tokenizers.md) — PaliGemma SentencePiece + FAST 完整算法。
