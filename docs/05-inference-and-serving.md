# 05 · 推理 pipeline 与远程部署

> 把 checkpoint 跑成机器人能调用的 50Hz policy。涉及：`Policy` 包装、JIT 编译、WebSocket 服务器、msgpack 协议、客户端实现、延迟优化。

---

## 1. 三种使用方式

```
方式 A：本地直接调用（单进程，最低延迟）
   Python script ─▶ policy_config.create_trained_policy(...) ─▶ Policy.infer(obs)

方式 B：本地 WebSocket（同机分离推理与控制进程，便于热重启 policy）
   Robot script ──WebSocket──▶ scripts/serve_policy.py

方式 C：远程 WebSocket（推理在云/远端 GPU，机器人端只需轻量 client）
   Robot ──远程网络──▶ openpi server （4090 或 H100）
```

A100/H100 端运行 server，机器人端只装 `openpi-client`（不需要 jax/torch）。

---

## 2. 从 checkpoint 到 Policy（核心入口）

`src/openpi/policies/policy_config.py`：

```python
def create_trained_policy(
    config: TrainConfig,
    checkpoint_dir: str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, NormStats] | None = None,
    is_pytorch: bool = False,
    pytorch_device: str = "cpu",
) -> Policy:
    # 1. 创建模型（架构）
    model = config.model.create(jax.random.key(0))
    # 2. 加载参数
    params = restore_params(checkpoint_dir / "params")
    nnx.update(model, params)
    # 3. 加载归一化统计（从 ckpt 内 assets/）
    norm_stats = norm_stats or load_norm_stats(...)
    # 4. 用 data_config 构造 transforms（同训练时一致）
    data_config = config.data.create(..., norm_stats=norm_stats)
    transforms_in  = [*repack.inputs, *data.inputs,  Normalize(norm_stats), *model.inputs]
    transforms_out = [*model.outputs,  Unnormalize(norm_stats), *data.outputs, *repack.outputs]
    # 5. 返回 Policy
    return Policy(model, transforms=transforms_in, output_transforms=transforms_out,
                  sample_kwargs=sample_kwargs, ...)
```

**关键设计**：训练和推理共享同一套 `data_config`（在 `config.py` 里定义），保证 transform 一致性。

---

## 3. `Policy.infer(obs)` 一次推理详解

`src/openpi/policies/policy.py:67`：

```python
def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
    # ① 复制（防止上游被 in-place 改）
    inputs = jax.tree.map(lambda x: x, obs)

    # ② 输入 transform（机器人坐标系 → π 内部空间 + 归一化 + tokenize）
    inputs = self._input_transform(inputs)

    # ③ 加 batch 轴 + 转成 jax.Array
    if not self._is_pytorch_model:
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
    else:
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        sample_rng = device

    # ④ 调用模型（已经被 module_jit 包过，第二次起无 host 开销）
    observation = Observation.from_dict(inputs)
    outputs = {
        "state": inputs["state"],
        "actions": self._sample_actions(sample_rng, observation, **sample_kwargs),
    }

    # ⑤ 去 batch 轴 + 转回 numpy
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    # ⑥ 输出 transform（反归一化 + π 内部空间 → 机器人坐标系）
    outputs = self._output_transform(outputs)

    # ⑦ 加上 timing 元数据
    outputs["policy_timing"] = {"infer_ms": (t2 - t1) * 1000}
    return outputs
```

**关于 `module_jit`**（`shared/nnx_utils.py`）：把 `model.sample_actions` 编译并缓存；第一次调用有 ~10s 编译时间，之后每次 < 30ms。

---

## 4. WebSocket Server

`src/openpi/serving/websocket_policy_server.py:15`：

```python
class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None): ...

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()

        # 一连接就发送 metadata（policy 配置）
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                # 接收 obs（dict，含 numpy array）
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # 推理
                action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": ..., "prev_total_ms": ...}

                # 回发 action
                await websocket.send(packer.pack(action))

            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                ...
```

**协议**（无 schema、纯 msgpack）：
```
client → server : msgpack.pack({"images": {...}, "state": ndarray, "prompt": "fold towel"})
server → client : msgpack.pack({"actions": ndarray[H, A], "state": ndarray, "policy_timing": {...}, "server_timing": {...}})
```

**健康检查**：`GET /healthz` → 200 OK（`websocket_policy_server.py:86`），便于 K8s liveness probe。

**为什么用 msgpack 而不是 protobuf / JSON？**
- numpy ndarray 在 msgpack-numpy 下零拷贝序列化（直接读 buffer）
- 不需要 schema，灵活适应不同机器人
- 比 protobuf 编码更紧凑（特别是大图像 bytes）

---

## 5. WebSocket Client（机器人端）

`packages/openpi-client/src/openpi_client/websocket_client_policy.py`（简化）：

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(
    host="10.0.0.5",  # server IP
    port=8000,
    metadata_callback=lambda m: print("Server metadata:", m),
)

while running:
    obs = {
        "images": {"cam_high": np.uint8 [3, 224, 224], ...},
        "state":  np.float32 [14],
        "prompt": "pour water into cup",
    }
    out = policy.infer(obs)
    actions = out["actions"]   # shape (50, 14)
    # 用 RTC 或简单的 chunk 截断执行 actions
```

**机器人端依赖**（仅 `openpi-client`）：
```toml
[dependencies]
"websockets",
"msgpack-numpy",
"numpy",
"openpi_client",
```
**不需要** jax / torch / transformers / paligemma 权重 → 机器人板 CPU 装得下。

---

## 6. 启动 server 的 CLI

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero \
  --policy.dir=checkpoints/pi0_libero/my_run/29000 \
  --port=8000 \
  --default-prompt="pick up the block and place it on the plate"
```

参数（来自 `scripts/serve_policy.py` + tyro）：
- `policy:checkpoint` — 从 ckpt 加载；也可 `policy:default` 用官方预训练
- `--policy.config` — 训练用的 TrainConfig 名（决定 model + transforms）
- `--policy.dir` — checkpoint 目录
- `--policy.is-pytorch` — 用 PyTorch 路径（默认 JAX）
- `--default-prompt` — obs 没带 prompt 时用这个

---

## 7. 推理延迟优化（生产经验）

### 7.1 编译缓存
```python
jax.config.update("jax_compilation_cache_dir", "~/.cache/jax")
```
首次启动 ~10-20s 编译；之后 reload 直接命中缓存（< 1s）。

### 7.2 KV cache
`pi0.py:233-237` 在 `sample_actions` 里：
- prefix（图像 + 语言）只前向一次，得到 KV cache
- 10 步去噪复用 KV cache，每步只算 suffix（state + 50 action token）

**节省**：约 80% 推理时间（prefix tokens 数 >> suffix tokens 数）。

### 7.3 bf16 推理
模型默认用 `bfloat16`（`pi0_config.py` 的 `dtype="bfloat16"`）。比 fp32 快 ~3-5x。

### 7.4 Action chunking + RTC
- 一次预测 50 步动作 → 机器人按 50Hz 控制 = 1 秒动作块
- 实际部署只执行前 25 步，剩下 25 步留给重叠（RTC 算法做平滑）
- 这让推理频率只需要 ~10Hz，而控制频率仍是 50Hz

### 7.5 PyTorch 路径的 torch.compile
`models_pytorch/pi0_pytorch.py` 用 `torch.compile(mode="reduce-overhead")` 预热。

---

## 8. 远程部署架构

```
┌─────────────────────────────┐
│  机器人端（边缘）            │
│  - Ubuntu 22.04 + ROS / Py  │
│  - openpi-client 唯一依赖    │
│  - 摄像头采集 + URDF 控制    │
└───────────┬─────────────────┘
            │ 局域网 / VPN / 直连云
            │ WebSocket（msgpack-numpy）
            ▼
┌─────────────────────────────┐
│  推理服务器（云端 / 桌面）   │
│  - Ubuntu + CUDA 12 + jax   │
│  - GPU: 4090 / A100 / H100  │
│  - scripts/serve_policy.py  │
│  - 多 worker：单 GPU 单 worker│
│  - K8s + Ingress + LB       │
└─────────────────────────────┘
```

**延迟拆解**（千兆局域网 / 同机房）：
- 图像编码 + msgpack pack：~3 ms（client 端）
- 网络往返：~1-5 ms（同机房）
- transform + tokenize：~2 ms（server CPU）
- prefix forward（首步）：~150 ms（一次性）
- suffix forward × 10：~30 ms（去噪环）
- output transform + msgpack pack：~2 ms
- **总：~40 ms（稳态）/ ~190 ms（首步）**

机器人端在等待 inference 时，按上一次的 action chunk 余量继续执行 → 看起来无延迟。

---

## 9. 常见部署坑

| 现象                                          | 原因                                                        | 解法                                                     |
| --------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| 推理第一帧很慢（~10s）                        | jit 编译                                                    | 预热：启动后跑一次 `policy.infer(dummy_obs)`             |
| 推理周期不稳定                                | XLA 重新编译（输入 shape 变了）                             | 确保 obs shape 固定；批量 size = 1                       |
| 显存爆                                        | 模型 + KV cache 占满                                        | 用 bf16；max_token_len 不要设太大                        |
| 机器人卡在 `await websocket.recv()`           | server 死循环 / 异常未发回                                  | 用 `process_request=_health_check` 做 liveness         |
| 远程机器人接收 action 太慢                    | msgpack pack 大 image，但 server 端实际只回 action          | 客户端 obs 用 `image_tools.resize_with_pad` 提前 resize  |
| `Connection closed` 频繁                      | 没设 `max_size=None`，超过 1MB 默认大小被拒                 | `_server.serve(..., max_size=None, compression=None)`   |
| Server 内存爆                                  | JAX 编译缓存累积                                            | 限制 `XLA_PYTHON_CLIENT_PREALLOCATE=false` + 重启周期    |

---

## 10. 一个端到端真机推理脚本（模板）

```python
# remote_inference_demo.py
import time
import numpy as np
from openpi_client import websocket_client_policy
from openpi_client import action_chunk_broker

# 1. 连接 server
policy = websocket_client_policy.WebsocketClientPolicy(host="10.0.0.5", port=8000)
print("Server metadata:", policy.get_server_metadata())

# 2. 用 action chunk broker 做 RTC / 简单截断
broker = action_chunk_broker.ActionChunkBroker(
    policy=policy,
    action_horizon=50,
    chunk_replan_steps=25,  # 每 25 步重新规划
)

# 3. 控制循环（50Hz）
hz = 50
period = 1.0 / hz
last_t = time.monotonic()
while True:
    obs = robot.read_obs()                       # {state, images, ...}
    obs["prompt"] = "pour water into cup"

    action = broker.infer(obs)["actions"][0]     # broker 帮你管 chunk
    robot.write_action(action)

    # 节拍同步
    elapsed = time.monotonic() - last_t
    if elapsed < period:
        time.sleep(period - elapsed)
    last_t = time.monotonic()
```

---

下一篇：[`06-data-transforms.md`](06-data-transforms.md) — 三层 transform pipeline + 每个机器人 policy 的实现细节。
