# 09 · Tokenizers 详解（PaliGemma SP + FAST）

> openpi 用两套 tokenizer：
> - **PaligemmaTokenizer**（SentencePiece）：把 prompt 编码成 token，π₀ 和 π₀.₅ 用
> - **FASTTokenizer**（FAST DCT+BPE）：把动作离散化成 token，π₀-FAST 用
>
> 这一篇把两者算法、源码、设计权衡讲透。

---

## 1. PaligemmaTokenizer（SentencePiece）

### 1.1 角色

把自然语言 prompt（"fold the towel"）+ π₀.₅ 时附加的 state 字符串 → token 序列 → 模型输入。

### 1.2 来源

PaliGemma 的官方 SentencePiece 模型：
```
gs://big_vision/paligemma_tokenizer.model
```
Vocab 大小 **257,152**（含 256 个图像 patch 占位 + 普通文本 + 1024 个 SigLIP token 占位 + 其他特殊 token）。

### 1.3 源码（`src/openpi/models/tokenizer.py:14`）

```python
class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")

        if state is not None:
            # π₀.₅ 格式：state 也编进去
            discretized_state = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # π₀ 格式：只有 prompt
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        # padding 到 max_len
        if len(tokens) < self._max_len:
            mask = [True] * len(tokens) + [False] * (self._max_len - len(tokens))
            tokens = tokens + [0] * (self._max_len - len(tokens))
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
```

### 1.4 SentencePiece 算法回顾

SentencePiece = **subword tokenization**（BPE 或 unigram LM）+ **直接对原始 bytes 编码**（不依赖 word segmentation）。
- 训练时统计 subword 频率，构建固定大小的 vocab
- 编码时贪心 / Viterbi 匹配最长 subword
- 解码时直接拼接（特殊 `▁` 表示空格）

```
"fold the towel"
→ ["▁fold", "▁the", "▁tow", "el"]
→ [12345, 6789, 23456, 7890]
```

### 1.5 π₀ vs π₀.₅ prompt 格式对比

```
π₀:   "<bos> fold the towel \n"
                                    ↑ 通过 LLM 双向 attention 给 action expert 提供条件

π₀.₅: "<bos> Task: fold the towel, State: 128 130 64 ... 200;\nAction: "
                                    ↑ state 用 256-bin 离散值字符串编码进 prompt
                                    ↑ 末尾 "Action: " 提示符
```

为什么 π₀.₅ 把 state 也编进文本？
- VLM 训练时见过的"任务-观察"模式 ≈ "看图 + 看状态描述 → 执行"
- 让 VLM 不只看图像还能"读"机器人当前状态 → 更好理解任务上下文
- 离散化避免连续值大尺度变化破坏 LLM 表征

---

## 2. FASTTokenizer（DCT + BPE 动作离散化）

### 2.1 角色

把连续动作 chunk `actions ∈ ℝ^{H × A}` 离散化为 token 序列，让 π₀-FAST 可以用纯 LLM 范式（next-token loss）训练。

### 2.2 算法概览

```
连续动作 (H, A) = (32, 32)
         │
         ▼ ① per-channel z-score（已经在 Normalize 那一步完成）
         ▼ ② DCT（离散余弦变换，每个 action dim 独立做）
         ▼ ③ 截断高频系数（保留前 K 个低频，K~8）
         ▼ ④ 量化到固定 grid（如 256 个 level）
         ▼ ⑤ BPE 编码（高频 pattern 合并成单 token）
离散 token 序列（典型 50-100 个 token）
```

**关键点**：
- DCT 把动作从时间域转到频域 → 低频系数捕获主要轨迹形状
- 截断高频 → 平滑轨迹（去除高频噪声），代价是丢失精细抖动
- 量化 + BPE → 把无限连续值映射到有限词表
- 反过来：解码时 BPE 还原系数序列 → 反 DCT → 连续动作

### 2.3 实现位置

FAST 算法本身在 HuggingFace 模型仓库 `physical-intelligence/fast`，通过 `AutoProcessor` 加载：

```python
self._fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
```

openpi 只用接口，不重写算法。

### 2.4 源码：`FASTTokenizer.tokenize`（tokenizer.py:64）

```python
def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None):
    cleaned_text = prompt.lower().strip().replace("_", " ")

    # 1. state 离散化（同 π₀.₅）
    discretized_state = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))

    # 2. prefix = "Task: ..., State: ...;\n"
    prefix = f"Task: {cleaned_text}, State: {state_str};\n"
    prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

    # 3. 训练时（actions 给定）：用 FAST tokenizer 编码 actions
    if actions is not None:
        action_tokens = self._fast_tokenizer(actions[None])[0]
        action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
        postfix_tokens = (
            self._paligemma_tokenizer.encode("Action: ")
            + action_tokens_in_pg.tolist()
            + self._paligemma_tokenizer.encode("|", add_eos=True)
        )
    else:
        postfix_tokens = []

    # 4. AR mask: 0 on prefix (bidirectional), 1 on postfix (causal)
    tokens = prefix_tokens + postfix_tokens
    ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
    loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # loss only on action tokens

    return tokens, mask, ar_mask, loss_mask
```

### 2.5 神奇的 vocab 复用：把 FAST token 映射到 PaliGemma 词表

```python
def _act_tokens_to_paligemma_tokens(self, tokens):
    # 把 FAST 输出的 0~N-1 整数映射成 PaliGemma vocab 末尾的 token id
    return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
```

**为什么**：
- PaliGemma vocab 末尾 128 个是特殊 token（不能用）
- 倒数第 129 个往前的若干个 token 几乎从未在文本训练中用过 → 复用给 FAST action token
- 这样 LLM forward / loss 完全等同标准 next-token prediction，**不需要修改模型架构**

### 2.6 推理时 decode

`tokenizer.py:119 extract_actions`：

```python
def extract_actions(self, tokens, action_horizon, action_dim):
    decoded = self._paligemma_tokenizer.decode(tokens.tolist())
    if "Action: " not in decoded:
        return np.zeros((action_horizon, action_dim))

    raw_action_tokens = np.array(self._paligemma_tokenizer.encode(
        decoded.split("Action: ")[1].split("|")[0].strip()
    ))
    action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
    return self._fast_tokenizer.decode([action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim)[0]
```

流程：
1. 把整段 token 序列 decode 回文本
2. 截取 "Action: " 和 "|" 之间的部分
3. 重新 encode 取数字 → 映射回 FAST token
4. `_fast_tokenizer.decode(...)` → 连续动作

### 2.7 FAST 算法的实质（更深层）

FAST 论文（<https://www.physicalintelligence.company/research/fast>）：

```
Input: actions ∈ ℝ^{H×A}, normalized to [-1, 1]
       (H=32, A=32 for π₀-FAST)

1. For each action dim d ∈ [0, A):
      coeffs[d] = DCT(actions[:, d])    # 时间维 DCT
      coeffs[d] = coeffs[d][:K]          # 截断到前 K 个低频系数（K=8 by default）

2. Flatten coeffs → 1D array of length A*K = 32*8 = 256 floats

3. Quantize each float to one of N=128 bins (uniform in [-1, 1])
   → integer array of length 256

4. Run BPE on this integer sequence (vocab ~1024)
   → token sequence of length 30-80（高频 pattern 合并）

5. Map to PaliGemma vocab IDs:
   token_id = paligemma_vocab_size - 1 - 128 - bpe_token
```

**优点**：
- 一个 32×32=1024 维的连续动作 chunk 压缩到 ~50 个 token
- 完全兼容标准 LLM 训练
- 训练时不需要 flow matching / diffusion 步骤

**缺点**：
- 推理慢：50-100 token 自回归 → 比 π₀ 的 10 步 ODE 慢
- 精度损失：DCT 截断 + 量化 = ~1% 动作误差

---

## 3. 第三种：BinningTokenizer / FSQTokenizer（RoboArena baseline）

`tokenizer.py:148 BinningTokenizer`：RT-2 / OpenVLA 风格，更朴素：
- 每个 action 值离散到 256 bin
- 不做 DCT，不做 BPE
- token 数 = `H × A`（很长）

openpi 中**只用于 RoboArena baseline 复现**，不是主流方案。

`fsq_tokenizer.py FSQTokenizer`：Finite Scalar Quantization 风格，更复杂的 baseline，同样仅用于 baseline。

---

## 4. 三种 tokenizer 对比

| 维度              | PaliGemma SP（语言）| FAST（动作 DCT+BPE）   | Binning（动作 quant）  |
| ----------------- | ------------------- | ---------------------- | ---------------------- |
| 输入              | 文本（任意字符）    | 动作 array (H, A)      | 动作 array (H, A)      |
| 输出 token 数     | 视文本长度 (~20-50) | ~30-80                 | H×A = 1024             |
| 算法              | SentencePiece BPE   | DCT + 量化 + BPE        | 直接量化               |
| 压缩率            | 高（subword）       | 高（频域 + BPE）       | 低                     |
| 重建精度          | 完全（无损）        | ~99%（DCT 截断 + 量化）| ~99.6%（仅量化）       |
| 计算开销          | 极低                | 低                     | 极低                   |
| 适用模型          | π₀, π₀.₅, π₀-FAST   | π₀-FAST                | baseline only          |

---

## 5. 如何在自己数据集上用 FAST

1. **训练阶段**：FAST tokenizer 是预训练好的（基于通用机器人动作语料）。多数情况下**不需要重训**，直接用 `physical-intelligence/fast`。

2. **如果你的动作分布非常特殊**（例如全是 quad-rotor 的 12 维 thrust），可以重训 FAST：
   ```bash
   pip install fast-tokenizer-tool
   fast-tokenizer-tool train --data my_actions.npy --output ./my_fast_tok
   ```
   然后在 `Pi0FASTConfig` 里 `fast_model_tokenizer = AutoProcessor.from_pretrained("./my_fast_tok")`。

3. **归一化必须**：FAST 假设输入 actions ∈ [-1, 1]，所以 `Normalize(use_quantiles=True)` 是前置条件。

---

## 6. Tokenizer 在 transform 管线中的位置

```
原始数据
   │
   ▼ repack_transforms      (字段重排)
   ▼ data_transforms        (机器人特定空间)
   ▼ Normalize              (z-score / quantile)
   ▼ model_transforms:
       ├─ ResizeImages
       ├─ TokenizePrompt (PaligemmaTokenizer)         ← π₀ / π₀.₅
       └─ TokenizeFASTInputs (FASTTokenizer)          ← π₀-FAST 替代上一个
   │
   ▼
模型输入：Observation(images, state, tokenized_prompt, tokenized_prompt_mask[, token_ar_mask, token_loss_mask])
```

---

## 7. 调试 tokenizer 的小工具

```python
from openpi.models.tokenizer import PaligemmaTokenizer, FASTTokenizer
import numpy as np

# 1. PaliGemma
tok = PaligemmaTokenizer(max_len=48)
tokens, mask = tok.tokenize("fold the towel", state=None)
print(tokens, "→", tok._tokenizer.decode(tokens[mask].tolist()))

# 2. π₀.₅ 格式
tokens, mask = tok.tokenize("fold the towel", state=np.random.uniform(-1, 1, 14))
print(tok._tokenizer.decode(tokens[mask].tolist()))

# 3. FAST
fast_tok = FASTTokenizer(max_len=256)
actions = np.random.uniform(-1, 1, (32, 32))
tokens, mask, ar_mask, loss_mask = fast_tok.tokenize(
    prompt="fold towel",
    state=np.random.uniform(-1, 1, 14),
    actions=actions,
)
print(f"Total tokens: {mask.sum()}, action tokens (loss=True): {loss_mask.sum()}")

# 4. 解码验证
extracted = fast_tok.extract_actions(tokens, action_horizon=32, action_dim=32)
print(f"Reconstruction MSE: {((actions - extracted)**2).mean():.4f}")
```

---

下一篇：[`10-finetune-cookbook.md`](10-finetune-cookbook.md) — 从零 fine-tune 自己的机器人。
