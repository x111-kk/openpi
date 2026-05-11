# 01 · 架构与数学原理（含 flow matching 推导）

> 目标：把 π₀ 的网络长什么样、为什么这么设计、flow matching 损失怎么推、attention mask 怎么排，全部讲透。
>
> 阅读完之后你应该能在脑子里画出 π₀ 的前向计算图。

---

## 1. 一图读懂 π₀ 架构

```
                                ┌───────────────────────────────────────┐
                                │      时间嵌入 τ ∈ [0,1]               │
                                │   sin-cos posemb → MLP                │
                                └───────────────────────────────────────┘
                                                │
                                                ▼ (条件)
   多视角 RGB         自然语言指令       proprio state          加噪 action chunk
   (224×224)        ("fold towel")        s_t ∈ ℝ³²            x_τ ∈ ℝ⁵⁰ˣ³²
        │                  │                     │                    │
        ▼                  ▼                     ▼                    ▼
   ┌──────────┐    ┌─────────────────┐    ┌──────────┐         ┌──────────┐
   │ SigLIP   │    │ SentencePiece   │    │ Linear   │         │ Linear   │
   │ ViT      │    │ Tokenizer       │    │ proj     │         │ proj     │
   │ So400m/14│    │ → embedding     │    │ → 1 token│         │ → 50 tok │
   └──────────┘    └─────────────────┘    └──────────┘         └──────────┘
        │                  │                     │                    │
        └─────┬────────────┘                     └──────┬─────────────┘
              │                                         │
              ▼ (prefix tokens, 全双向 attention)        ▼ (suffix tokens, 因果 attention)
   ╔════════════════════════════════╗         ╔══════════════════════════╗
   ║  Expert 1: PaliGemma LLM 2B    ║◀───────▶║ Expert 2: Action 300M   ║
   ║  (width=2048, head_dim=256)    ║ 共享    ║ (width=1024)            ║
   ║  18 个 Block × (Attn + FFN)    ║ attn    ║ 18 个 Block × (Attn+FFN)║
   ╚════════════════════════════════╝         ╚══════════════════════════╝
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │ Linear out   │
                                                   │ → v_t ∈ ℝ⁵⁰ˣ³² │
                                                   │ 速度场       │
                                                   └──────────────┘

prefix mask:  [图像 tok × N] [语言 tok × M]              ← 全双向（mask_ar=0）
suffix mask:  [state tok]   [action tok × 50]            ← 第一个 ar=1 块开始因果
                            └─────────────┘
                            内部全双向（mask_ar=0）
```

**核心创新**（必须记住）：

1. **MoE（Mixture-of-Experts）双塔共享 attention**——两个不同大小、不同权重的 Transformer 共享 KV，让 VLM 的视觉/语言知识直接条件化 action expert。
2. **Action expert 没有视觉/语言路径**——它的输入是 (state, noisy_action, time)，但它的 Q/K/V 与 VLM 的 KV 拼接在一起算 attention。
3. **混合 attention mask**（prefix-LM + causal）——prefix 块（图像+语言）全双向；suffix 块（state, actions）开始因果。
4. **flow matching 头**——不像 LLM 输出 logits，而是输出**速度场** v(x, τ)，配 ODE 积分采样动作。

---

## 2. Flow Matching 数学推导（从 0 到代码）

### 2.1 目标

给定专家动作 `a* ∈ ℝᴴˣᴬ`（H=horizon, A=action_dim），我们想训一个生成模型 `p_θ(a | obs)`。

**Diffusion 思路**：从噪声 `ε ∼ N(0, I)` 逐步降噪到 `a*`。

**Flow matching 思路**：定义一个 **条件向量场** `u_t(x | a*)`，让粒子沿它流动，在 t=0 是数据，t=1 是噪声。训一个模型 `v_θ` 去匹配这个向量场，推理时反向 ODE。

### 2.2 选择"直线路径"（rectified flow / OT path）

最简单的选择：直线插值
$$
x_t = (1 - t) \cdot a^* + t \cdot \varepsilon, \quad t \in [0, 1]
$$

对 t 求导得到**条件向量场**：
$$
u_t(x_t | a^*) = \frac{d x_t}{dt} = \varepsilon - a^*
$$

也就是说，"从数据走到噪声"的瞬时速度就是噪声减数据。**注意：openpi 用的就是这个约定，t=1 是噪声、t=0 是数据**（和 π₀ 论文反过来，源码 `pi0.py:226` 也加了注释道歉）。

### 2.3 Flow Matching 损失

模型 `v_θ(x_t, t, obs)` 直接回归条件向量场，MSE 损失：
$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim p(t),\, \varepsilon, \, a^*} \left[ \| v_\theta(x_t, t, \text{obs}) - (\varepsilon - a^*) \|^2 \right]
$$

### 2.4 时间分布 p(t)

openpi 用 **Beta(1.5, 1)** 分布并 clip 到 (0.001, 1.0)：

```python
# src/openpi/models/pi0.py:197
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
```

Beta(1.5, 1) 的 PDF 在 t=1 附近（高噪声端）权重更大，强迫模型在难的去噪步上多练。

### 2.5 推理时的 ODE 求解

训练完后，模型 `v_θ` 学到了速度场。推理时反向 Euler 步：
$$
x_{t - \Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, \text{obs})
$$

从 `x_1 ∼ N(0, I)` 出发，10 步内积分到 `x_0`（数据）：

```python
# src/openpi/models/pi0.py:228
dt = -1.0 / num_steps   # num_steps 默认 10
def step(carry):
    x_t, time = carry
    ...
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    return x_t + dt * v_t, time + dt
```

**为什么 10 步就够？** 因为是直线路径（rectified flow），单步残差小；加上 action expert 只 300M，10 步前向 ~30ms，刚好够 50Hz 机器人控制。

### 2.6 推理"小技巧"——KV cache

`obs`（图像/语言）在 10 步去噪里都不变，所以**先跑一次 prefix 拿到 KV cache**，10 次 suffix 前向都复用：

```python
# src/openpi/models/pi0.py:234
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

def step(carry):
    ...
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens], mask=..., kv_cache=kv_cache, ...
    )
```

实际部署时延 = `1 × prefix_forward + 10 × suffix_forward`，suffix 前向只有 50 个 action token，非常快。

---

## 3. 双塔 MoE Attention 详解（源码级）

### 3.1 整体思想

`gemma.Module` 是个 transformer，它的特别之处是**接收一个 `Sequence[Config]` 而不是单个 config**——每个 config 对应一个 expert，所有 expert 共享 attention 但有独立的 Q/K/V/FFN 权重。

### 3.2 关键源码逐行

`src/openpi/models/gemma.py:158-260` 的 `Attention.__call__`：

```python
def __call__(self, xs, positions, attn_mask, kv_cache):
    # xs: List[Optional[Array]]  每个 expert 一个 token 序列
    # 所有 expert 必须 head_dim/num_heads/num_kv_heads 一致 → KV 可以拼接

    qkvs = []
    for i, (x, config) in enumerate(zip(xs, self.configs)):
        if x is None: continue
        # 每个 expert 用自己的 Q/K/V einsum（权重独立！）
        qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))

    # 关键：所有 expert 的 Q/K/V 沿序列轴拼接！共享 attention！
    q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs))

    q = _apply_rope(q, positions=positions)
    k = _apply_rope(k, positions=positions)

    # GQA: q grouped on K head
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

    masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
    probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
    encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

    # 拼接的输出再切回各 expert，用各自的 out projection
    out = []
    start = 0
    for i, (x, config) in enumerate(zip(xs, self.configs)):
        if x is not None:
            end = start + x.shape[1]
            out_einsum = lora.Einsum(...)
            out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
            start = end
    return out
```

**重点**：
- Q/K/V 拼接的本质 = "两个 expert 看一张共享的全局序列，但每个 expert 用自己的投影"。
- 因为 Q 也是拼接的，所以 expert A 的 token 可以 attend 到 expert B 的 token（取决于 mask）。
- VLM expert 永远不 attend 到 action expert（mask 决定）。

### 3.3 Attention Mask（prefix-LM + causal）

mask 由 `make_attn_mask` 函数生成，输入是 `input_mask`（padding mask）和 `mask_ar`（每 token 是否开启因果）：

```python
# src/openpi/models/pi0.py:19
def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

`mask_ar` 的设计很巧妙：
- `[0, 0, 0, 0]` → 全双向（prefix-LM）
- `[1, 1, 1, 1]` → 全因果
- `[0, 0, 1, 1]` → 前面双向、后面因果，开启因果的 token 不能被前面看到

在 π₀ 中：
- 图像 token 之间双向（`ar_mask += [False] * N`）
- 语言 token 与图像之间全双向（同上）
- state token 第一个 ar=1（一段新的因果块，看不到 action）
- action token 第一个 ar=1，内部全双向（`ar_mask += [True] + [False] * (H-1)`）

这一段在 `src/openpi/models/pi0.py:106-186` 的 `embed_prefix` 和 `embed_suffix`。

---

## 4. AdaRMS（π₀.₅ 的时间条件）

π₀ 把时间 t 拼到 action token 里（concat 后 MLP）。π₀.₅ 改用 **AdaRMS**——让 RMSNorm 的 scale/shift/gate 由时间 MLP 输出条件化：

```python
# src/openpi/models/gemma.py:127
modulation = nn.Dense(x.shape[-1] * 3, kernel_init=zeros, dtype=dtype)(cond)
scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
normed_inputs = normed_inputs * (1 + scale) + shift
return normed_inputs.astype(dtype), gate
```

- `scale, shift` 用于 RMSNorm 的输出调制（FiLM 风格）
- `gate` 用于 residual 上的门控（`x + gate * Attn(norm(x))`）
- 全部初始化为 0 → 训练初期等同于不加条件，稳定收敛
- 仅在 action expert 启用（`use_adarms=[False, True]`，见 `pi0.py:80`）

**意义**：时间信号能影响每一层的归一化和 residual，比单纯拼到 token 表达力更强；同时 VLM expert 完全不受时间扰动 → 配合 KI 训练，VLM 表征不会因为 action loss 漂移。

---

## 5. RoPE / GQA / RMSNorm 速览

- **RoPE**（Rotary Position Embedding）：见 `gemma.py:69`，对 Q/K 旋转，不增参数，外推友好。
- **GQA**（Grouped Query Attention）：`num_heads > num_kv_heads`，多组 Q 共享同一组 KV，减少 KV cache 大小。Gemma 2B 配置 `num_heads=8, num_kv_heads=1`（极端 GQA，1 个 KV 头）。
- **RMSNorm**：见 `gemma.py:100-131`，比 LayerNorm 少一个减均值，常数项；scale 学习；在 fp32 计算后再 cast 回。

---

## 6. FFN（gated FFN，类 SwiGLU 但用 GELU）

`lora.FeedForward`（`src/openpi/models/lora.py:88`）：

```python
gate_value = nn.gelu(x @ W_gate)    # gated
hidden     = x @ W_up               # up projection
hidden     = gate_value * hidden    # gate × up
output     = hidden @ W_down        # down projection
```

每层 FFN 都可加 LoRA（`lora_config` 参数控制，详见 `07-key-techniques.md`）。

---

## 7. SigLIP 视觉塔（So400m/14）

`src/openpi/models/siglip.py`：
- 输入：`224×224×3` 图像，patch size 14 → 16×16=256 个 patch token
- ViT 编码：27 层 transformer, width=1152
- 输出：`pool_type='none'` → 直接吐 256 个 patch embedding，不做 cls/mean pool
- 投影：`Linear(1152 → 2048)`（PaliGemma 的 width）

一帧图像 = 256 个 prefix token。三视角 ALOHA = 768 token；多视角 + 语言 prompt ~1000 token 量级。

---

## 8. 数学公式速查表

| 概念              | 公式                                                             |
| ----------------- | ---------------------------------------------------------------- |
| 直线路径插值      | `x_t = (1-t) a* + t ε`                                           |
| 条件向量场        | `u_t = ε - a*`                                                   |
| Flow matching loss| `L = E[‖v_θ(x_t, t, obs) - (ε - a*)‖²]`                        |
| 时间采样          | `t ~ Beta(1.5, 1) * 0.999 + 0.001`                              |
| 反向 ODE          | `x_{t-Δt} = x_t - Δt · v_θ(x_t, t, obs)`                        |
| Attention 注意    | 共享 K/V/位置；不同 expert 独立 Q/K/V/FFN 权重                   |
| AdaRMS            | `out = (1+scale) · RMS(x) + shift`，gate 用于 residual           |

---

下一篇：[`02-code-framework-map.md`](02-code-framework-map.md) — 把每个 .py 文件按职责分类、给出关键类/函数的行号锚点。
