# 07 · 关键技术点源码精读

> 把 openpi 里几个最高 ROI 的工程亮点逐个拆开：LoRA、AdaRMS、KV cache、gemma_fast 推理优化、EMA、`nnx.scan`、混合精度。

---

## 1. LoRA（Low-Rank Adaptation）

`src/openpi/models/lora.py`，148 行的极简实现，但功能完整：可对任意 einsum 加 LoRA、自动推 LoRA einsum 等式、支持 rank-stabilized LoRA。

### 1.1 原理回顾

主权重 W ∈ ℝ^{d_in × d_out}。LoRA 不直接训 W，而是训两个低秩矩阵 A ∈ ℝ^{d_in × r}, B ∈ ℝ^{r × d_out}，让：

```
y = x W + α/r · x A B
```

训练时 W 冻结（bf16 即可），只更新 A, B（fp32）。r 通常 8–64，参数量减少 ~100×。

### 1.2 `LoRAConfig`（lora.py:11）

```python
@struct.dataclass
class LoRAConfig:
    rank: int                      # r（低秩维度）
    alpha: float = 1.0             # 缩放系数（影响 LR 实效）
    init_fn: Initializer = normal(stddev=0.01)
    rslora: bool = False           # rank-stabilized：scaling = α/√r 而非 α/r
    axes: tuple[int, int] = (-2, -1)   # einsum 中哪两个轴拆 LoRA
    label: str = "L"               # einsum 里新插入的 rank 维度的字母

    @property
    def scaling_value(self):
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
```

### 1.3 LoRA Einsum 的妙处（lora.py:33）

```python
class Einsum(nn.Module):
    shape: tuple[int, ...]
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", init_fn, self.shape)
        if config := self.lora_config:
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank      # 在最后一轴改成 rank
            shape_b[config.axes[0]] = config.rank      # 在倒数第二轴改成 rank
            self.w_a = self.param("lora_a", init_fn, shape_a)
            self.w_b = self.param("lora_b", init_fn, shape_b)

    def __call__(self, eqn, x):
        result = jnp.einsum(eqn, x, self.w.astype(dtype))
        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a)
            lora = jnp.einsum(eqn_b, lora, self.w_b)
            result = result + lora * config.scaling_value
        return result
```

### 1.4 自动推 LoRA einsum 等式（lora.py:67）

例：QKV 投影 `BSD,3KDH->3BSKH`（B=batch, S=seq, D=d_model, K=num_kv_heads, H=head_dim, 3 = q/k/v 三个）。
- 假设 `axes=(-2, -1)` → A 的形状把 H 改成 L → `(3,K,D,L)`，B 的形状把 D 改成 L → `(3,K,L,H)`
- 自动算出：
  - `eqn_a = "BSD,3KDL->3BSKL"` （x · A）
  - `eqn_b = "3BSKL,3KLH->3BSKH"` （上一步 · B）
- 加上 scaling：`result += (x A) B · α/r`

**漂亮的地方**：不管 einsum 多复杂，只要给定 `axes` 就能自动 derive，无需手写 PyTorch 风格 `nn.Linear` 拆 LoRA。

### 1.5 LoRA Feed Forward（lora.py:88）

FFN 也加 LoRA，三个矩阵都拆（gate / up / down）。

### 1.6 在 π₀ 中的使用

`gemma.py:88-117` 的 `gemma_2b_lora` / `gemma_300m_lora` 配置：
```python
lora_configs = {
    "attn": LoRAConfig(rank=16, alpha=16.0),   # attention 加 LoRA
    "ffn":  LoRAConfig(rank=16, alpha=16.0),   # FFN 加 LoRA
}
```

`pi0_config.py` 里选 variant：
```python
paligemma_variant: Variant = "gemma_2b_lora"     # VLM 加 LoRA
action_expert_variant: Variant = "gemma_300m"    # action expert 全量训（小，加 LoRA 没必要）
```

`weight_loaders.py:52` 的 `CheckpointWeightLoader._merge_params` 用 regex `.*lora.*` 在加载基础 ckpt 时**自动补齐缺失的 LoRA 权重**（zeros 初始化）。

### 1.7 LoRA 实战建议

- **rank 16, alpha 16** 是 π 推荐配置（scaling = 1.0）
- 学习率 `5e-5` ~ `1e-4`（比全量微调高 10×）
- 训完可以 merge：`W_merged = W + α/r · A B`，部署时无额外开销
- 显存：约为全量训练的 1/3（不存 optimizer state for frozen params）

---

## 2. AdaRMS（自适应 RMSNorm，π₀.₅ 引入）

`src/openpi/models/gemma.py:114-156` 的 `RMSNorm`：

```python
class RMSNorm(nn.Module):
    adarms: bool = False        # 是否启用 AdaRMS
    adarms_cond_dim: int = 256  # 时间条件 MLP 输出维度

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        scale = self.param("scale", zeros, (x.shape[-1]))
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + 1e-6)
        normed = normed * (1 + scale)

        gate = None
        if self.adarms:
            assert cond is not None
            modulation = nn.Dense(x.shape[-1] * 3, kernel_init=zeros, dtype=dtype)(cond)
            scale_, shift_, gate_ = jnp.split(modulation[:, None, :], 3, axis=-1)
            normed = normed * (1 + scale_) + shift_         # FiLM 风格调制
            gate = gate_

        return normed.astype(dtype), gate
```

### 2.1 在 Block 中的使用

`gemma.py:300` 的 `Block.__call__`：

```python
def __call__(self, x, cond, ...):
    # 1. 第一个 RMSNorm + Attention
    normed, gate1 = self.pre_attn_norm(x, cond)
    attn_out = self.attn(normed, ...)
    if gate1 is not None:
        attn_out = attn_out * (1 + gate1)                   # 门控 residual
    x = x + attn_out

    # 2. 第二个 RMSNorm + FFN
    normed, gate2 = self.pre_ffw_norm(x, cond)
    ffn_out = self.mlp(normed)
    if gate2 is not None:
        ffn_out = ffn_out * (1 + gate2)
    x = x + ffn_out

    return x
```

### 2.2 为什么有效

- **传统**：时间 t concat 到 token embedding 后过一层 transformer。时间影响所有层是间接的、靠 attention 传递。
- **AdaRMS**：时间通过 **每层** 的 RMSNorm scale/shift 和 residual gate 直接调制。每一层都"看见"时间。
- **零初始化**：scale/shift/gate 的输出 MLP 用 zeros_init → 训练初期等同 vanilla RMSNorm → 不破坏预训练。
- **仅 action expert 启用**：`use_adarms=[False, True]`（pi0.py:80）→ VLM expert 不受时间扰动 → 配合 Knowledge Insulation 训练。

### 2.3 实测效果

π₀.₅ 论文报告：相同数据下 AdaRMS 比 token-concat 时间条件 **降低 ~15% MSE 损失**，泛化（unseen language tasks）更好。

---

## 3. KV cache（推理加速核心）

### 3.1 为什么需要

π₀ 推理时跑 10 步 ODE。如果每步都重算所有 token 的 KV，整个 prefix（图像+语言）就被算 10 次，浪费 ~9× 计算。

### 3.2 实现位置

`src/openpi/models/pi0.py:217 sample_actions`：

```python
def sample_actions(self, rng, observation, *, num_steps=10):
    observation = preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    # 第一次前向：只过 prefix → 得到 KV cache
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    # 反向 ODE 10 步，复用 kv_cache
    dt = -1.0 / num_steps
    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        # mask 拼接（prefix + suffix）
        ...
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],         # prefix 传 None → 走 cache
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return x_t + dt * v_t, time + dt

    x_t, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_t
```

### 3.3 mask 拼接的细节

第一步前向：只看 prefix。
后续步：suffix 的 token 要能 attend prefix（双向）+ attend suffix 自己（依 ar_mask）。

`pi0.py:240-265` 拼了完整 attention mask：

```python
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)        # [B, P, P]
suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)        # [B, S, S]
# suffix → prefix 全双向
prefix_attn_mask_repeated = einops.repeat(prefix_attn_mask, "b p1 p2 -> b (s + p1) p2", s=S)  # 简化伪代码
...
```

实际代码用 `jnp.concatenate` 沿 axis=-1 拼出 `[B, S, P+S]`，然后让 cross-attention 可见性正确。

### 3.4 性能

| 阶段              | tokens | 时间（H100, bf16） |
| ----------------- | ------ | ------------------ |
| Prefix forward    | ~1024  | ~150 ms            |
| Suffix × 10 steps | 50 × 10| ~30 ms             |
| **总**            | -      | **~180 ms**        |

如果不缓存（每步全 forward），总耗时会到 ~1.5 s。**KV cache 是把 π₀ 推理变实时的关键**。

---

## 4. `gemma_fast.py` —— 自回归推理优化

`src/openpi/models/gemma_fast.py`（437 行）是给 π₀-FAST 用的 Gemma 变体，专为**自回归解码**优化：

| 区别              | gemma.py（用于 π₀）                                | gemma_fast.py（用于 π₀-FAST）           |
| ----------------- | --------------------------------------------------- | ---------------------------------------- |
| 支持多 expert     | 是（MoE 双塔）                                      | 否（单塔）                               |
| KV cache 类型     | dict 临时构造                                       | 显式 ring buffer，预分配空间             |
| 解码方式          | 一次 forward 全 sequence                            | 每 step 只 forward 1 token，逐 token decode |
| AdaRMS            | 支持                                                | 不需要（FAST 不用时间）                   |
| 主要使用者        | `pi0.py`                                            | `pi0_fast.py`                            |

### 4.1 自回归解码循环

`pi0_fast.py:sample_actions`：

```python
def sample_actions(self, rng, observation, *, max_decoding_steps=128, ...):
    # 1. encode 输入 prefix
    tokens, mask = self.tokenizer.tokenize_inference(observation)
    initial_logits, kv_cache = self.gemma_fast.forward(tokens, kv_cache=None)

    # 2. 自回归循环
    cur_token = sample_token(initial_logits[:, -1, :])
    out_tokens = [cur_token]
    for step in range(max_decoding_steps):
        logits, kv_cache = self.gemma_fast.forward(cur_token, kv_cache=kv_cache)
        cur_token = sample_token(logits[:, -1, :])
        if cur_token == EOS_TOKEN: break
        out_tokens.append(cur_token)

    # 3. 用 FAST tokenizer 反解出 actions
    actions = self.fast_tokenizer.decode(out_tokens)
    return actions
```

详见 [`09-tokenizers.md`](09-tokenizers.md)。

---

## 5. EMA（Exponential Moving Average）

`scripts/train.py:169` train_step 里：

```python
if state.ema_decay is not None:
    new_state.ema_params = jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
        state.ema_params,
        new_params,
    )
```

`ema_decay=0.99`（`TrainConfig` 默认）。推理时用 `ema_params` 而不是当前 step 的 params，更稳定。

实际部署：ckpt 保存时 `_split_params(state)` 会把 ema_params 单独存到 `params/`，推理直接加载这个目录。

---

## 6. `nn.scan` —— 用 scan 折叠 18 层 Block

`gemma.py:359` 附近：

```python
if config.scan:
    block_apply = nn.scan(
        Block,
        variable_axes={"params": 0},     # 第 0 维是层数
        split_rngs={"params": True},
        length=config.depth,
    )
    x = block_apply(...)(x, ...)
else:
    for _ in range(config.depth):
        x = Block(...)(x, ...)
```

**好处**：
- 编译时只需展开 1 层，剩下 17 层由 `scan` 处理 → JIT 编译速度快 18×
- 显存少（不需要保存 18 层独立 params 的副本）
- 支持 remat（gradient checkpointing），节省训练显存

### 6.1 Remat

```python
remat_policy="nothing_saveable"
```
在 Block 上用 `jax.checkpoint(policy=...)` → 反向传播时重算激活，省显存。

---

## 7. 混合精度策略

| 数据                          | dtype     | 说明                                                            |
| ----------------------------- | --------- | --------------------------------------------------------------- |
| 模型权重（可训）              | fp32      | optimizer state 仍是 fp32                                       |
| 模型权重（冻结）              | bf16      | LoRA 模式下冻结的主干 bf16，省显存                              |
| activation                    | bf16      | 默认                                                            |
| RMSNorm 的 var/mean 计算       | fp32      | 数值稳定性需要                                                  |
| AdamW 内部 m, v                | bf16/fp32 | optax 默认 fp32                                                 |
| flow matching loss            | fp32      | 累积之前先 cast                                                 |

`pi0_config.py`：
```python
dtype: str = "bfloat16"
```

`gemma.py:RMSNorm` 内部：
```python
var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
normed = x * jax.lax.rsqrt(var + 1e-6)
...
return normed.astype(dtype)   # 转回 bf16
```

---

## 8. 时间嵌入（sin-cos posemb）

`pi0.py:48`：

```python
def posemb_sincos(pos, embedding_dim, min_period=4e-3, max_period=4.0):
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = pos[:, None] / period[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)
```

为什么 `min_period=4e-3`？因为 t ∈ [0, 1]，最细粒度需要分辨 ~1/(num_steps) 量级，10 步意味着 ~0.1 间隔；冗余覆盖到 4e-3 就稳了。

---

## 9. 一张速查表

| 技术       | 文件:行号                       | 适用场景                                |
| ---------- | ------------------------------- | --------------------------------------- |
| LoRA       | `lora.py`                       | 显存吃紧、单机微调                      |
| AdaRMS     | `gemma.py:114-156`              | π₀.₅+（条件 norm）                      |
| KV cache   | `pi0.py:217-279`                | 推理（必须开）                          |
| nn.scan    | `gemma.py:359`                  | 训练编译速度优化                        |
| EMA        | `scripts/train.py:169-175`      | 推理稳定性                              |
| 混合精度   | 全局                            | 训练/推理显存                           |
| gemma_fast | `gemma_fast.py`                 | π₀-FAST 自回归推理                      |
| FSDP       | `sharding.py:48-102`            | 多卡训练                                |

---

下一篇：[`08-version-source-diff.md`](08-version-source-diff.md) — π₀ vs π₀-FAST vs π₀.₅ vs π₀.₆ 源码逐项 diff。
