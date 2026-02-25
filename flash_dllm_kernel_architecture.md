# flash-dLLM: A Modular CUDA Kernel Architecture for Diffusion Language Models

## 1. 设计哲学：为什么 dLLM 需要自己的 "Flash Attention"

### Flash Attention 成功的核心原因

Flash Attention 之所以能成为 AR 模型的基础设施，是因为它抓住了一个 **通用且基本的操作**：

```
Output = Softmax(QK^T / √d + Mask) × V
```

无论是 GPT、LLaMA、Mistral 还是 Gemma，attention 的计算本质都是这个公式。Flash Attention 通过 IO-aware tiling 将其从 O(N²) 内存降到 O(N)，所有 AR 模型都受益。

### dLLM 的 "Flash Attention 等价物" 是什么？

通过分析 10+ 个 dLLM 框架（LLaDA、MDLM、SEDD、Dream、BD3-LM、D2F、Fast-dLLM、WeDLM、DAWN、Focus-dLLM、dInfer、SGLang），我们发现**所有框架共享同一个推理管线**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    dLLM 通用 Denoising Step                          │
│                                                                      │
│  1. [MASK_STATE]   确定哪些位置是 MASK（噪声调度 / 重新 mask）         │
│  2. [ATTENTION]    用适当的 mask 模式运行 transformer                  │
│  3. [KV_CACHE]     选择性复用/刷新 KV cache                           │
│  4. [PREDICT]      获取 MASK 位置的 logits                            │
│  5. [CONFIDENCE]   计算每个位置的置信度                                │
│  6. [SELECT]       排序并选择要 commit 的 token                       │
│  7. [UPDATE]       写入 committed token，remask 其余                  │
│                                                                      │
│  步骤 2-3 占主导计算量（attention + KV 管理）                         │
│  步骤 4-7 占主导 kernel launch 开销（多个小操作）                     │
└──────────────────────────────────────────────────────────────────────┘
```

**关键洞察**：就像 Flash Attention 是 AR 模型中 `Softmax(QK^T)V` 的通用加速一样，dLLM 需要的是对这个 **7 步管线中每个环节的基本原语** 进行加速，然后让它们像乐高积木一样自由组合。

---

## 2. 跨框架共性分析

### 2.1 所有 dLLM 框架的 Attention 模式分类

| 模式 | 描述 | 使用者 |
|------|------|--------|
| `FULL_BIDIR` | 全双向（无 mask） | LLaDA v1, MDLM, SEDD, Dream, DICE |
| `BLOCK_CAUSAL` | Block 内双向 + Block 间因果 | LLaDA2.X, BD3-LM, D2F, Fast-dLLM v2 |
| `BLOCK_DIAG` | 严格块对角（只看自己 block） | D2F 的某些变体, 训练中的 noisy-noisy 部分 |
| `COMPOSITE` | 训练时 [x_t; x_0] 的 4 部分复合 mask | BD3-LM, Fast-dLLM v2 训练 |
| `CAUSAL_REORDER` | 拓扑重排后的标准因果 | WeDLM |
| `SPARSE_BIDIR` | 动态稀疏双向 | Focus-dLLM, DAWN |

**趋势**：纯双向模型 → block-causal 是整个生态的收敛方向（因为 KV cache 兼容性）。

### 2.2 所有框架共享的 4 个核心操作

```
                    ┌─────────────────┐
                    │  flash_dllm     │
                    │  Kernel Library │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
    │ Attention  │     │ Denoise   │     │ Training  │
    │ Primitives │     │ Primitives│     │ Primitives│
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                  │                  │
   ┌──────┼──────┐    ┌─────┼─────┐    ┌──────┼──────┐
   │      │      │    │     │     │    │      │      │
   BC   COMP   SEL    CS   CU    BM   MCE   NOISE  CMASK
```

**BC** = Block-Causal Attn, **COMP** = Composite Mask, **SEL** = Selective KV Refresh
**CS** = Confidence-Select, **CU** = Commit-Update, **BM** = Block Manager
**MCE** = Masked Cross-Entropy, **NOISE** = Noise Application, **CMASK** = Complementary Masking

---

## 3. 核心 Kernel 设计

### Level 0: Attention 原语 (训练 + 推理通用)

#### Kernel 0a: `flash_attn_block_causal` — 最高优先级

**这是 dLLM 生态的 "Flash Attention"**。所有 block diffusion 模型的基础操作。

```
设计原则：不是新操作，而是 flash_attn 的新 mask 模式
```

**接口设计**：

```python
def flash_attn_block_causal(
    q, k, v,                          # [batch, seqlen, heads, dim]
    block_size: int = 32,             # dLLM block 大小
    softmax_scale: float = None,
    causal_within_block: bool = False, # True=block 内也 causal, False=block 内双向
    # 以下继承 flash_attn 现有参数
    window_size: Tuple[int,int] = (-1,-1),
    softcap: float = 0.0,
    return_softmax_lse: bool = False,
) -> Tensor:
```

**CUDA 实现要点**：

```cpp
// 在 flash-attention 的 Hopper Mask struct 中添加:
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma>
struct Mask {
    // 新增 block diffusion 参数
    int const block_size;              // dLLM block 大小
    bool const bidir_within_block;     // block 内是否双向

    // 核心 mask 判定 — O(1) 算术运算，无需读内存
    CUTLASS_DEVICE
    bool is_block_causal_attended(int q_idx, int k_idx) const {
        int q_block = q_idx / block_size;   // 整数除法 → FastDivmod
        int k_block = k_idx / block_size;
        if (k_block > q_block) return false;        // 因果：不看后面的 block
        if (k_block < q_block) return true;          // 看所有前面的 block
        // 同一个 block 内
        return bidir_within_block ? true : (q_idx >= k_idx);  // 双向或因果
    }

    // Tile 级优化 — 整个 tile 可以跳过
    CUTLASS_DEVICE
    bool can_skip_tile(int q_tile_start, int k_tile_start, int tile_size) const {
        int q_block_min = q_tile_start / block_size;
        int k_block_max = (k_tile_start + tile_size - 1) / block_size;
        return k_block_max > q_block_min;  // 整个 tile 都在未来 block → 跳过
    }

    // 整个 tile 都不需要 per-element masking
    CUTLASS_DEVICE
    bool tile_is_full(int q_tile_start, int k_tile_start, int tile_size) const {
        int q_block_min = q_tile_start / block_size;
        int q_block_max = (q_tile_start + tile_size - 1) / block_size;
        int k_block_max = (k_tile_start + tile_size - 1) / block_size;
        // 所有 K 都在所有 Q 的 block 之前（且不在同一 block 边界上）
        return k_block_max < q_block_min;
    }
};
```

**三级 tile 分类**（与 FlexAttention 的 BlockSparseTensors 对应）：

```
对于每个 (q_tile, kv_tile) 对:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. SKIP (k_block_max > q_block_min)  → 完全不计算              │
  │ 2. FULL (k_block_max < q_block_min)  → 无 mask，直接计算       │
  │ 3. PARTIAL (边界 tile)               → per-element mask 判定   │
  └─────────────────────────────────────────────────────────────────┘

可视化（block_size=32, tile_size=64）:

     kv_tile_0  kv_tile_1  kv_tile_2  kv_tile_3
q_0  [PARTIAL   SKIP       SKIP       SKIP     ]
q_1  [FULL      PARTIAL    SKIP       SKIP     ]
q_2  [FULL      FULL       PARTIAL    SKIP     ]
q_3  [FULL      FULL       FULL       PARTIAL  ]

SKIP tiles: ~50% (上三角) → 节省 50% 计算
FULL tiles: ~25% → 无 mask 开销
PARTIAL tiles: ~25% → 需要 per-element block_size 判定
```

**与 `attention_chunk` 的关系**：在现有 Hopper mask.h 的 `attention_chunk_divmod` 路径中，mask 逻辑是：

```cpp
// 现有 attention_chunk: 严格块对角
col_limit_left = round_down(chunk_divmod, row_idx) - thread_offset;
col_limit_right = col_limit_left + chunk_divmod.divisor;
```

Block-causal 只需将 `col_limit_left` 改为 0（允许看所有前面的 chunk）：

```cpp
// Block-causal: 修改为
if (bidir_within_block) {
    col_limit_left = 0;  // 可以看所有前面的 block
    col_limit_right = round_down(chunk_divmod, row_idx) + chunk_divmod.divisor;
} else {
    col_limit_left = 0;
    col_limit_right = row_idx + 1;  // 标准 causal 到当前位置
}
```

**这是改动最小、收益最大的修改** — 本质上是把 `attention_chunk` 从 "严格块对角" 扩展为 "块级下三角"。

#### Kernel 0b: `flash_attn_composite_mask` — 训练用复合 Mask

训练时的 `[x_t; x_0]` 拼接需要 4 部分复合 mask（见上一篇分析的 2.4 节）。

**接口设计**：

```python
def flash_attn_composite_mask(
    q, k, v,                          # [batch, 2*seqlen, heads, dim]
    block_size: int = 32,
    seq_len: int,                      # 原始序列长度（不含拼接）
    # 第一半是 noisy, 第二半是 clean
    mask_mode: str = "bdlm_train",     # 预定义的复合 mask 模式
) -> Tensor:
```

**CUDA mask 判定**：

```cpp
CUTLASS_DEVICE
bool composite_mask(int q_idx, int k_idx, int seq_len, int block_size) const {
    bool q_is_noisy = (q_idx < seq_len);
    bool k_is_noisy = (k_idx < seq_len);
    int q_block = q_is_noisy ? (q_idx / block_size) : ((q_idx - seq_len) / block_size);
    int k_block = k_is_noisy ? (k_idx / block_size) : ((k_idx - seq_len) / block_size);

    if (q_is_noisy && k_is_noisy) {
        return q_block == k_block;           // M_BD: 同 block 内双向
    }
    if (q_is_noisy && !k_is_noisy) {
        return q_block > k_block;            // M_OBC: noisy 看更早的 clean
    }
    if (!q_is_noisy && !k_is_noisy) {
        return q_block >= k_block;           // M_BC: clean 内 block-causal
    }
    return false;                             // clean 不看 noisy
}
```

**Tile 级跳过分析**：复合 mask 的结构非常规整——整个上半部分（clean→noisy）是全 0，可以跳过 ~25% 的 tiles。

#### Kernel 0c: `flash_attn_selective_kv` — 选择性 KV Refresh (DualCache)

所有推理框架的核心瓶颈：在 denoising 迭代中如何复用 frozen block 的 KV。

**背景**：Fast-dLLM 的 DualCache 使用 `replace_position`（一个 `[batch, seq_len]` 的 0/1 mask）
在已有 KV cache 中**原地覆盖**指定位置的 KV，而非 AR 模型的 append 模式。当前实现是
Python for 循环 + 逐 batch scatter，本身就是重要的优化目标：

```python
# Fast-dLLM v1 的 replace_position 实现（llada/model/modeling_llada.py）:
if replace_position is None:
    k = torch.cat((past_key, k), dim=-2)           # 标准 AR: append
else:
    for batch_idx in range(B):                       # Python for 循环!
        indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
        past_key[batch_idx, :, indices] = k[batch_idx, :, :len(indices)]    # scatter
        past_value[batch_idx, :, indices] = v[batch_idx, :, :len(indices)]
    k, v = past_key, past_value
```

Fast-dLLM v2 使用两层缓存：
- **Block-level cache（精确）**：block-causal attention 使已解码 block 的 KV 精确有效
- **Sub-block cache / DualCache（近似）**：block 内 denoising 迭代时复用 prefix+suffix KV

**接口设计**：

```python
def flash_attn_selective_kv(
    q,                                 # [batch, active_len, heads, dim]  — 当前 block 的 Q
    kv_cache,                          # [batch, total_len, 2, kv_heads, dim]  — 全局 cache
    active_kv,                         # [batch, active_len, 2, kv_heads, dim] — 当前 block 新 KV
    cache_seqlens,                     # [batch] — cache 中有效 KV 的长度
    replace_position: Tensor,          # [batch, total_len] bool — DualCache 核心：指定覆盖位置
    block_size: int,                   # block 大小
    block_causal: bool = True,         # v2=True (精确 block 间 cache), v1=False (近似)
    vicinity_window: int = 0,          # vicinity refresh 时刷新邻近多少个位置
) -> Tuple[Tensor, Tensor]:            # (attn_output, updated_kv_cache)
```

**核心优化**：

```
场景：Block 5 正在 denoising（32 步），前 4 个 block 已 frozen

标准方法（LLaDA 当前）:
  每步：Q=[block5], K=V=[block1,block2,block3,block4,block5]  全部重算
  32 步 × 5 blocks × 32 tokens = 5120 tokens 的 attention，32 次

flash_attn_selective_kv:
  第 0 步：全序列 forward，cache 所有 KV
  第 1-31 步：Q=[block5 的 32 tokens], K=V=[cached 128 tokens + block5 新 32 tokens]
  只需要 160 tokens 的 attention，31 次

计算节省：31 × (5120 - 160) / (32 × 5120) ≈ 94%
```

**关键设计点**：
- `replace_mode="overwrite"`: Fast-dLLM 的 DualCache 模式，直接覆盖当前 block 的 KV
- `replace_mode="vicinity_refresh"`: dInfer 的做法，同时刷新邻近 block 的 KV
- 内部使用 flash_attn_with_kvcache 的 paged attention 机制

---

### Level 1: Denoising 原语 (推理阶段通用)

#### Kernel 1a: `fused_confidence_select_remask` — 融合置信度选择

**所有 dLLM 推理框架都执行这个操作**，目前需要 4-6 次 kernel launch。

**当前 Python 实现（LLaDA2.1）**：

```python
# 6 个独立操作，6 次 kernel launch
probs = F.softmax(logits / temperature, dim=-1)          # 1. temperature + softmax
x0 = torch.multinomial(probs, 1)                         # 2. sampling
x0_p = torch.gather(probs, -1, x0)                       # 3. confidence 提取
confidence = torch.where(mask, x0_p, -inf)               # 4. mask 过滤
high_conf = confidence > threshold                         # 5. threshold 比较
# top-k fallback if insufficient                          # 6. top-k selection
x[transfer_idx] = x0[transfer_idx]                        # 7. scatter update
```

**融合 kernel 设计**：

```cpp
__global__ void fused_confidence_select_remask(
    const float* __restrict__ logits,    // [batch, block_len, vocab_size]
    int*         __restrict__ sequence,  // [batch, block_len] — in-place update
    const bool*  __restrict__ mask,      // [batch, block_len] — which positions are MASK
    const int    mask_token_id,
    const float  temperature,
    const float  threshold,
    const int    min_transfer,           // 至少 unmask 多少个
    const int    top_k,
    const float  top_p,
    // outputs
    float*       __restrict__ confidence_out,  // [batch, block_len]
    int*         __restrict__ num_transferred  // [batch]
) {
    // 每个 thread block 处理一个 position
    int pos = blockIdx.x;
    int batch = blockIdx.y;

    if (!mask[batch * block_len + pos]) return;  // 跳过非 MASK 位置

    // Step 1: Temperature + Online Softmax (不需要物化整个 prob 分布)
    float max_logit = -INFINITY;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        max_logit = fmaxf(max_logit, logits[...] / temperature);
    }
    // warp reduce max_logit...

    // Step 2: Top-k filtering + Sampling (在 shared memory 中)
    // 使用 approximate top-k (bucket sort) 避免全排序

    // Step 3: Confidence = sampled token 的 probability
    float conf = sampled_prob;

    // Step 4: 写入 confidence（供后续 block-level top-k 决策）
    confidence_out[batch * block_len + pos] = conf;

    // Step 5: 如果 confidence > threshold，直接 commit
    if (conf > threshold) {
        sequence[batch * block_len + pos] = sampled_token;
        atomicAdd(&num_transferred[batch], 1);
    }
}

// 第二个 kernel（仅在 threshold 模式不足时）: top-k fallback
__global__ void topk_fallback_commit(
    const float* confidence, int* sequence, int* sampled_tokens,
    int min_transfer, int* num_transferred
) {
    // 如果 num_transferred < min_transfer，选择 top-k confidence 的位置
    // 使用 bitonic sort 在 shared memory 中排序
}
```

**性能估算**：
- 当前 6 次 kernel launch → 1-2 次
- 消除中间 tensor 分配（probs, x0, x0_p, confidence, high_conf, transfer_idx）
- 预计 block_length=32 时，denoising step 的非 attention 部分加速 **3-5x**

#### Kernel 1b: `fused_token_editing` — Token 编辑（LLaDA2.1 专用但通用化）

LLaDA2.1 的 token editing 和 dInfer 的 credit decoding 都有类似模式：对**已解码位置**进行修正。

```python
def fused_token_editing(
    logits,                    # [batch, block_len, vocab_size]
    current_tokens,            # [batch, block_len]
    mask_positions,            # [batch, block_len] bool
    prompt_positions,          # [batch, block_len] bool
    editing_threshold: float,
    temperature: float,
) -> Tuple[Tensor, Tensor]:   # (updated_tokens, edit_mask)
```

这个 kernel 融合了：
1. 对非 MASK、非 prompt 位置采样新 token
2. 比较新 token 与旧 token 是否不同
3. 检查新 token 的置信度是否超过 editing_threshold
4. 只在三个条件同时满足时更新

---

### Level 2: 训练原语

#### Kernel 2a: `fused_masked_cross_entropy` — 融合 Masked 交叉熵

**所有 dLLM 训练框架的核心 loss 计算**。

```python
def fused_masked_cross_entropy(
    logits,                    # [batch, seq_len, vocab_size]
    targets,                   # [batch, seq_len]
    mask,                      # [batch, seq_len] — 只在 masked 位置计算 loss
    t,                         # [batch] — timestep (用于 ELBO 权重)
    weight_fn: str = "1/t",   # "1/t" | "clipped" | "uniform"
) -> Tensor:                   # scalar loss
```

**优化点**：
- 跳过 `mask=False` 的位置（当 t 很小时，~99% 的位置可以跳过）
- 融合 softmax + CE + 1/t 权重 + reduction
- 避免物化 `[batch, seq_len, vocab_size]` 的中间 logits

```cpp
// 核心思路：只对 masked positions 计算 CE
__global__ void fused_masked_ce(
    const float* logits, const int* targets, const bool* mask,
    const float* t, float* loss, int batch, int seq_len, int vocab_size
) {
    // 每个 thread block 处理一个 (batch, position)
    int b = blockIdx.y, pos = blockIdx.x;
    if (!mask[b * seq_len + pos]) return;  // 跳过非 MASK 位置

    // Online softmax + CE for this position
    float target_logit = logits[b * seq_len * vocab_size + pos * vocab_size + targets[...]];
    float log_sum_exp = online_log_sum_exp(logits + offset, vocab_size);
    float ce = log_sum_exp - target_logit;

    // Apply ELBO weight: 1/t
    float weight = 1.0f / t[b];
    atomicAdd(loss, ce * weight);
}
```

**预计加速**：当 masking rate < 50% 时（常见于小 t），节省 50%+ 计算。

#### Kernel 2b: `apply_discrete_noise` — 融合噪声注入

```python
def apply_discrete_noise(
    clean_tokens,              # [batch, seq_len]
    t,                         # [batch] — masking rate
    mask_token_id: int,
    noise_type: str = "absorbing",  # "absorbing" | "uniform" | "custom"
    complementary: bool = False,    # 是否同时生成互补 mask
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    # Returns: (noisy_tokens, mask, optional complementary_noisy_tokens)
```

当前实现：`torch.rand() < t` → `torch.where()` = 2+ kernel launches
融合后：1 kernel launch，同时生成 noisy tokens 和 mask

---

### Level 3: 调度与协调原语 (系统级)

#### Kernel 3a: `block_denoising_scheduler` — Block 级调度器

管理多个 block 在不同 denoising 阶段的协调：

```python
class BlockDenoisingScheduler:
    """管理 block-sequential + within-block parallel 的混合调度"""

    def __init__(self, num_blocks, block_size, steps_per_block):
        self.block_states = [PENDING] * num_blocks  # PENDING/ACTIVE/FROZEN
        self.step_counters = [0] * num_blocks
        self.confidence_history = []

    def get_next_batch(self) -> List[BlockTask]:
        """返回下一批要执行的 block 任务"""
        # 支持多种策略:
        # 1. Sequential: 一次一个 block
        # 2. Pipeline: 多个 block 重叠（D2F 风格）
        # 3. Speculative: 提前开始未来 block（需要验证）

    def commit_block(self, block_idx, tokens, confidence):
        """标记 block 完成，冻结其 KV cache"""

    def should_refresh_kv(self, block_idx) -> bool:
        """判断是否需要 vicinity refresh（dInfer 风格）"""
```

---

## 4. 组合模式：如何用基本 Kernel 构建完整系统

### 模式 A: LLaDA2.X 推理 (Block-Causal + Confidence)

```python
# 用到的 kernel: 0a + 0c + 1a
for block_idx in range(num_blocks):
    # 第 0 步: prefill
    kv_cache = flash_attn_block_causal(full_q, full_k, full_v, block_size=32)

    for step in range(steps):
        # attention with cached prefix
        output = flash_attn_selective_kv(
            q=active_q, kv_cache=kv_cache, active_kv=active_kv,
            block_start=block_idx*32, replace_mode="overwrite"
        )
        # fused denoising step
        fused_confidence_select_remask(
            logits=output.logits, sequence=x, mask=mask_positions,
            threshold=0.95, temperature=0.0
        )
```

### 模式 B: BD3-LM / Fast-dLLM v2 训练 (Composite Mask + Masked CE)

```python
# 用到的 kernel: 0b + 2a + 2b
noisy_x, mask = apply_discrete_noise(x_0, t, mask_id=MASK)
x_train = torch.cat([noisy_x, x_0], dim=1)  # [batch, 2L]

# 一次 forward 训练所有 block
output = flash_attn_composite_mask(q, k, v, block_size=32, seq_len=L)

# 融合 masked CE loss
loss = fused_masked_cross_entropy(output.logits, x_0, mask, t, weight_fn="1/t")
```

### 模式 C: Fast-dLLM v1 推理 (Approximate KV Cache)

```python
# 用到的 kernel: 0c (vicinity refresh mode) + 1a
for block_idx in range(num_blocks):
    # 第 0 步: full forward, populate cache
    output = model.forward(x, use_cache=True)
    kv_cache = output.kv_cache

    for step in range(steps):
        # 只处理当前 block，vicinity refresh 邻近 block
        output = flash_attn_selective_kv(
            q=block_q, kv_cache=kv_cache, active_kv=block_kv,
            replace_mode="vicinity_refresh", vicinity_window=32
        )
        fused_confidence_select_remask(output.logits, ...)
```

### 模式 D: WeDLM 推理 (拓扑重排 → 标准 causal)

```python
# WeDLM 的独特之处：通过拓扑排序将 bidir 转为 causal
# 用到的 kernel: 标准 flash_attn (causal=True) + 1a
reorder_indices = topological_sort(dependency_graph)
x_reordered = x[reorder_indices]

# 标准 causal attention — 直接用现有 flash_attn！
output = flash_attn_func(q, k, v, causal=True)

# 恢复原始顺序后做 confidence select
fused_confidence_select_remask(output.logits[inverse_reorder], ...)
```

---

## 5. 与 FlexAttention 的关系：互补而非竞争

### FlexAttention 的优势

FlexAttention 的 `mask_mod` 函数式 API 非常优雅：

```python
def block_causal_mask(b, h, q_idx, kv_idx):
    return (q_idx // block_size) >= (kv_idx // block_size)

block_mask = create_block_mask(block_causal_mask, B, H, Q_LEN, KV_LEN)
output = flex_attention(q, k, v, block_mask=block_mask)
```

CuTe DSL 版本在 Hopper 上达到 Flash Attention-3 的 **95%** 性能。

### 我们的 kernel 提供什么额外价值

1. **Fused Denoising Primitives**：FlexAttention 只加速 attention 本身，不涉及 confidence-select-remask 等 dLLM 特有操作。我们的 Level 1 kernel 填补这个空白。

2. **KV Cache 集成**：FlexAttention 没有原生的 KV cache 支持。我们的 `flash_attn_selective_kv` 将 attention + cache management 融合。

3. **Pattern-Specific Fast Path**：对于 block-causal 这个已知模式，专用 kernel 可以做编译期优化（`FastDivmod` 替代运行时除法、常量 block_size 模板参数），比 FlexAttention 的通用路径快 5-15%。

### 推荐架构

```
User API (Python)
       │
       ├── FlexAttention-style mask_mod → 通用路径 (研究/实验)
       │      │
       │      └── create_block_mask → BlockSparseTensors
       │                │
       │                └── CuTe DSL generic kernel (95% of FA3)
       │
       └── flash_dllm named patterns → 快速路径 (生产部署)
              │
              ├── "block_causal"    → flash_attn_block_causal kernel
              ├── "composite_train" → flash_attn_composite_mask kernel
              ├── "selective_kv"    → flash_attn_selective_kv kernel
              └── "auto"           → pattern detection → 选择最优路径
```

---

## 6. 实现路线图

### Phase 1: 基础 Attention（最高 ROI）
- `flash_attn_block_causal` — 在 Hopper mask.h 中添加 block-causal 模式
- 预计改动量：~200 行 CUDA + ~100 行 Python binding
- 预计影响：所有 block diffusion 模型推理加速 2-4x

### Phase 2: KV Cache 集成
- `flash_attn_selective_kv` — 基于 flash_attn_with_kvcache 扩展
- 预计改动量：~500 行 CUDA
- 预计影响：denoising 循环加速 3-8x

### Phase 3: 融合 Denoising 操作
- `fused_confidence_select_remask` — 新 CUDA kernel
- `fused_masked_cross_entropy` — 新 CUDA kernel
- 预计改动量：~1000 行 CUDA
- 预计影响：非 attention 操作加速 3-5x

### Phase 4: 训练优化
- `flash_attn_composite_mask` — 复合 mask 支持
- `apply_discrete_noise` — 融合噪声注入
- 预计改动量：~300 行 CUDA + ~200 行 Python
- 预计影响：训练 throughput 提升 20-40%

### Phase 5: 系统级优化
- Block denoising scheduler
- Inter-block pipeline parallelism
- Dynamic block-size adaptation

---

## 7. 总结：dLLM 的 "Flash Attention 时刻"

Flash Attention 的成功在于它抓住了一个 **所有 AR 模型共享的基本操作**（causal attention），并提供了一个 **通用的高性能实现**。

对于 dLLM 生态，等价的机会是：

```
AR 的 Flash Attention      ←→     dLLM 的 flash_dllm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
causal attention           ←→     block-causal attention
KV cache (append-only)     ←→     selective KV cache (replace + refresh)
next-token sampling        ←→     confidence-select-remask
CE loss                    ←→     masked CE with ELBO weight
─────────────────────────────────────────────────────
1 个核心 kernel            ←→     ~6 个可组合的 kernel 原语
```

**核心洞察**：dLLM 不需要一个 monolithic mega-kernel，而是需要 **一组可组合的基本原语**，就像 BLAS 提供 Level 1/2/3 操作一样。每个原语足够通用以服务所有 dLLM 框架，同时足够专用以实现接近理论峰值的性能。

这就是 **flash-dLLM** 的设计目标：成为 diffusion language model 生态的标准 CUDA 基础设施。
