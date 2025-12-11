
# 核心思想
FlashAttention 的核心在于减少 GPU HBM (显存) 的读写次数。标准 Attention 需要计算并存储 $N \times N$ 的巨大矩阵，而 FlashAttention 通过分块计算，将计算过程限制在快速的 SRAM 中。
# 前向传播

已知量：Q, K, V -> [N, D];

待求量：O -> [N, D]; M, LSE -> [N,]

计算公式：

$O = P * V$ -> [N, D] = [N, N] * [N, D]

$P = Softmax(S)$;  

$S = \frac{Q * K^T}{\sqrt{D}}$

Online Softmax:

$P = \frac{exp(S-M)}{\sum{exp{S-M}}}$


基本计算流 (The "Fixed Q, Flowing K/V" Strategy):

并行维度：Grid 主要按 $N$ 维度（Sequence Length）划分。每个 Program ID (PID) 负责计算一个 BLOCK_SIZE_QO 大小的 Q 块。

Triton中的grid有两个维度，第一个维度为Q的BLOCK ID，第二个维度为Head ID：

```python
grid = lambda META: (
    triton.cdiv(N, META['BLOCK_SIZE_QO']),
    B * H
)
```

驻留 (Resident)：当前 PID 加载对应的 Q 块 到 SRAM 中。在整个 Inner Loop 过程中，这个 Q 块保持不变。

流动 (Scanning)：K 块 和 V 块 从 HBM 中被分批加载到 SRAM（滑动窗口）。

计算：Q 块与当前的 K 块计算分数，更新 Softmax，与 V 块相乘，累加到输出 O 中，然后丢弃当前的 K/V，加载下一块。

关于MASK：Q只能看到它和它之前的K，对于小于Q的K来说，不需要MASK，对于和Q有重叠的K来说，这个单独的BLOCK里需要MASK。

比如现在的Q BLOCK是Q的64~127行，那么0~63的K不需要MASK,64~127的K需要MASK。

具体实现是根据是否需要MASK，来改变K和V的下界和上界。比如我现在需要计算64~127的Q块，如果需要MASK，那么就只加载[64, 127]的K和V，并且进行mask处理：

```python
causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
S += tl.where(causal_mask, 0, -1.0e6)
```

如果不需要MASK，那么就只加载，[0, 63]的K和V，这时直接计算，不需要mask处理，至于128及以上的，直接不加载。

具体流程：

根据是否需要mask，确定K和V的加载上下界。然后遍历界内：

1. 固定在SRAM中的是$Q_i%，现在遍历到的是$K_j$和$V_j$，计算$S_{ij} = Q_i * K_j^T$ # shape: [BLOCK_SIZE_QO, BLOCK_SIZE_KV]

2. 更新全局最大值，$M_{new} = max(max(S_{ij}), M_{old})$ # shape: [BLOCK_SIZE_QO]

3. 计算修正因子, $\alpha=exp(M_{old} - M_{new})$

4. 计算P，$P = exp(S - M_{new})$

5. 更新分母和LSE, $L_{new} = sum(P, axis=1)$, $LSE = LSE * \alpha + L_{new}$

6. 更新输出O， $O = O * \alpha + P * V$

7. 更新全局最大值，$M = M_{new}$

8. 移动K和V的指针

代码：

```python
# 遍历K和V的分块
    for start_kv in range(low, high, BLOCK_SIZE_KV):
        # 编译器优化提示
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # KV的行的mask
        mask_KV_N = offsets_KV_N < N

        # 加载K_T，注意因为是转置，所以mask也要转置，mask_KV_N[None, :]
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)

        # Q和K_T相乘
        S = tl.dot(Q, K_T) * scale
        
        if DIAGONAL:
            # offsets_QO_N:[8, 9, 10, 11]
            # offsets_KV_N:[8, 9, 10, 11]
            # 0, 1, 1, 1
            # 0, 0, 1, 1
            # 0, 0, 0, 1
            # 0, 0, 0, 0 
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            S += tl.where(causal_mask, 0, -1.0e6)
        
        # Online Softmax
        # # 需要不断更新每一行的Max值
        # # [BLOCK_SIZE_QO]
        # m_cur_block = tl.max(S, axis=1)

        # # 更新全局最大值
        # # [BLOCK_SIZE_QO]
        # M_new = tl.maximum(m_cur_block, M)
        
        M_new = tl.maximum(M, tl.max(S, axis=1))
        
        # 修正S并计算P
        # P = exp(S - M_new)
        S -= M_new[:, None]
        P = tl.exp2(S)
        
        # 更新分母和L
        # [BLOCK_SIZE_QO]
        L_new = tl.sum(P, axis=1)
        # 根据新的max值，计算修正系数
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new
        
        # 更新O的输出
        # O_new = O_old * alpha + P @ V
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
        O = O * alpha[:, None]
        O = tl.dot(P, V, acc=O)
        
        # 更新M [BLOCK_SIZE_QO]
        M = M_new
        
        # 移动指针到下一个Block
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
```
# 后向传播

## 公式推导
前向传播的公式为：

$S = scale * (Q * K^T)$

$P = softmax(S) = \frac{exp(S-M)}{\sum{exp(S-M)}}$

$O = P * V$

1. 推导$dV$

$dV = P^T * dO$

2. 推导$dS$

$dP = dO * V^T$

$$
对于一行S=\{s_1, s_2, s_3, s_N\},S_i的梯度dS_{i} = \sum_{j}dP_j * \frac{\partial P_j}{\partial S_i} \\


当i=j时，\frac{\partial P_j}{\partial S_i} = P_j * (1-P_j) \\
当i\neq j时，\frac{\partial P_j}{\partial S_i}  = -P_j*P_i \\
所以，dS_i = \sum_{j} dP_j * (-P_j*P_i)+dP_i*P_i*(1-P_i)=\sum_{j} dP_j * (-P_j*P_i) + dP_i * dP_j \\
= P_i * (dP_i-\sum_{i} dP_j * dP_j)
$$
接下来求$\sum_{i} dP_j * dP_j$

$$
对于第i行的Q，\Delta _{i}=\sum_{k} dP_{ik} * dP_{ik} \\
dP_{ik} = dO_i * V_k^T = \sum _{d} dO_{id} * V_{dk}^T = \sum _{d} dO_{id} * V_{kd} \\
\therefore \Delta _{i}=\sum_{k} P_{ik} \sum_{d}dO_{id}V_{kd}=\sum_{d}dO_{id}\sum_{k}P_{ik}V_{kd} (求和的交换律) \\
=\sum_{d}dO_{id}*O_{id}
\therefore \Delta_{i} = sum(dO * o, axis=-1) \\
\therefore dS = P \circ (dP - \Delta), 其中 \circ 表示逐元素乘法 (Element-wise multiplication)。
$$

3. 推导$dQ, dK$

$$
根据公式S = scale * (Q * K^T)可求得\\
dQ = dS \cdot K \cdot scale\\
dK = dS^T \cdot Q \cdot scale

$$

## Triton实现流程

1. 预处理 (Preprocessing Kernel)

+ 目的：计算辅助变量 $\Delta$。

+ 输入：$O, dO$

+ 操作：Delta = sum(O * dO, axis=-1)

+ 输出：$\Delta$ (Shape: [B, H, N])

2. 反向传播的主逻辑

### 固定Q，流动K和V

此时grid按照K进行分块（BLOCK_SIZE_KV），每个kernel处理一个BLOCK的K和V，把该BLOCK加载到SRAM中，流动Q、dO、LSE、Delta。

首先要重新计算$S = QK^T$ [BLOCK_SIZE_Q, D] * [D, BLOCK_SIZE_KV] = [BLOCK_SIZE_Q, BLOCK_SIZE_KV]

然后计算$P=exp(S-M)$

$dV += P^T \cdot dO$   

shape: [BLOCK_SIZE_KV, BLOCK_SIZE_Q] * [BLOCK_SIZE_Q, D] = [BLOCK_SIZE_KV, D]

$dK += dS^T \cdot Q (dS = P (dP - \Delta))$ 

shape:  [BLOCK_SIZE_KV, BLOCK_SIZE_Q] * [BLOCK_SIZE_Q, D] = [BLOCK_SIZE_KV, D]

### 固定Q，流动K和V

此时grid按照Q进行分块（BLOCK_SIZE_Q），每个kernel处理一个BLOCK的Q，把该BLOCK的Q以及dO,LSE,Delta加载到SRAM中，流动K、V.

重计算P

累积梯度： $dQ += dS \cdot K$

import torch
import torch.nn as nn
import numpy as np
row = torch.arange(8, 12)
col = torch.arange(8, 12)

row[:, None] < col[None, :]
a = torch.randn((5, 3))
a
a.stride(0)
a.stride(1)
row_offsets = torch.arange(0, 3)
col_offsets = torch.arange(0, 2)

row_offsets[:, None] * a.stride(0) + col_offsets[None, :] * a.stride(1)
row_offsets[None, :] * a.stride(0) + col_offsets[:, None] * a.stride(1)
a = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)


a = range(start=0, stop=1, step=1)
a
import torch
import triton
import triton.language as tl
import math
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

from google.colab import drive
drive.mount('/content/drive')

@triton.jit
def _attn_fwd_inner(
    # input data
    Q, O, L, M, # Q是固定的当前块，K和V是滑动窗口，O和L和M是中间结果
    K_ptr, V_ptr, # K和V的全图
    K_T_offsets, V_offsets, # 初始偏移量
    
    # index
    block_index_QO, # 当前Q处理的是第几个block
    scale,
    stride_K_N, stride_V_N,
    
    # constant configure
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr, 
    DIAGONAL: tl.constexpr, # 是否是对角注意力
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    N: tl.constexpr, Dh: tl.constexpr
):
    # 如果当前是计算对角线上的，比如计算P_22这个块，这个块的大小为4*4，那么这个小块里
    # 的下三角才会计算
    if DIAGONAL:
        low = block_index_QO * BLOCK_SIZE_QO
        high = (block_index_QO + 1) * BLOCK_SIZE_QO
        # 让编译器知道low可以被BLOCK_SIZE_QO整除
        low = tl.multiple_of(low, BLOCK_SIZE_QO)
    # 如果计算的是对角线以下的，那么直接计算就行，不用mask
    else:
        # KV的选取从0开始，到当前Q块的起始位置结束，不考虑当前Q块及其以后的KV块
        low, high = 0, block_index_QO * BLOCK_SIZE_QO
    
    # 根据low和high，将K和V的指针推进到正确的地方
    # [Dh, BLOCK_SIZE_KV]
    K_T_offsets += low * stride_K_N
    V_offsets += low * stride_V_N
    # 当前处理的KV块的行索引 [0, 1, 2, 3, ...] + low
    offsets_KV_N += low
    
    # 遍历K和V的分块
    for start_kv in range(low, high, BLOCK_SIZE_KV):
        # 编译器优化提示
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # KV的行的mask
        mask_KV_N = offsets_KV_N < N

        # 加载K_T，注意因为是转置，所以mask也要转置，mask_KV_N[None, :]
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)

        # Q和K_T相乘
        S = tl.dot(Q, K_T) * scale
        
        if DIAGONAL:
            # offsets_QO_N:[8, 9, 10, 11]
            # offsets_KV_N:[8, 9, 10, 11]
            # 0, 1, 1, 1
            # 0, 0, 1, 1
            # 0, 0, 0, 1
            # 0, 0, 0, 0 
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            S += tl.where(causal_mask, 0, -1.0e6)
        
        # Online Softmax
        # # 需要不断更新每一行的Max值
        # # [BLOCK_SIZE_QO]
        # m_cur_block = tl.max(S, axis=1)

        # # 更新全局最大值
        # # [BLOCK_SIZE_QO]
        # M_new = tl.maximum(m_cur_block, M)
        
        M_new = tl.maximum(M, tl.max(S, axis=1))
        
        # 修正S并计算P
        # P = exp(S - M_new)
        S -= M_new[:, None]
        P = tl.exp2(S)
        
        # 更新分母和L
        # [BLOCK_SIZE_QO]
        L_new = tl.sum(P, axis=1)
        # 根据新的max值，计算修正系数
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new
        
        # 更新O的输出
        # O_new = O_old * alpha + P @ V
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
        O = O * alpha[:, None]
        O = tl.dot(P, V, acc=O)
        
        # 更新M [BLOCK_SIZE_QO]
        M = M_new
        
        # 移动指针到下一个Block
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return M, L, O
        
# -----------------------------------------------------------
# Forward Pass (前向传播) Kernel
# -----------------------------------------------------------
@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16]#, 32, 64, 128]
        for BLOCK_SIZE_KV in [16]#, 32, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["Dh"],
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr,
    LSE_ptr,
    scale,
    
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, H: tl.constexpr, N: tl.constexpr, Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr
):
    rln2: tl.constexpr = 1.4426950408889634  # log2(e)
    scale *= rln2
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
    # Q和O的行索引
    block_index_QO = tl.program_id(0)
    
    # 计算batch和head的索引
    index_BH = tl.program_id(1)
    
    # 计算batch和head的索引
    index_B = index_BH // H
    index_H = index_BH % H
    
    # Q和O的起始位置（batch和head维度）
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H
        
    # 计算行和列的offset
    # range(0, BLOCK_SIZE_QO) + block_start
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    # range(0, BLOCK_SIZE_KV)
    offset_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    # range(0, Dh)
    offset_Dh = tl.arange(0, Dh)
    
    # Q的偏移矩阵[BLOCK_SIZE_QO, Dh]
    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offset_Dh[None, :] * stride_Q_Dh
    
    # K的偏移矩阵[Dh, BLOCK_SIZE_KV]，这里是转置存储的
    K_T_offsets = offset_Dh[:, None] * stride_K_Dh + offset_KV_N[None, :] * stride_K_N
    
    # V的偏移矩阵[BLOCK_SIZE_KV, Dh]
    V_offsets = offset_KV_N[:, None] * stride_V_N + offset_Dh[None, :] * stride_V_Dh
    
    # 读取Q到SRAM
    
    # mask_QO_N: [BLOCK_SIZE_QO]
    mask_QO_N = offsets_QO_N < N
    
    # tl.load的mask参数需要和要load的数据shape一致，所以这里加一个维度
    # 将Q加载到SRAM中
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0)
    
    # 初始化中间结果
    # 每一行的最大值
    M = tl.full(shape=(BLOCK_SIZE_QO,), value=-1e6, dtype=tl.float32)
    
    # 每一行的sum exp
    L = tl.full(shape=(BLOCK_SIZE_QO,), value=1.0, dtype=tl.float32)
    
    # 输出O的初始化
    O = tl.zeros(shape=(BLOCK_SIZE_QO, Dh), dtype=tl.float32)
    
    # 处理对角线以下的块，DIAGONAL=False
    M, L, O = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False,
        offsets_QO_N, offset_KV_N,
        N, Dh,
    )
    
    # 处理对角线上的块，DIAGONAL=True
    M, L, O = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True,
        offsets_QO_N, offset_KV_N,
        N, Dh,
    )
    
    # 计算完整的softmax
    O = O / L[:, None]
    
    # 计算Log sum exp，为了backward
    # LSE = log(L) + M
    # L = exp2(sum(P) - M)
    LSE = tl.math.log2(L) + M
    
    # 将计算结果从SRAM写回HBM
    # 1. 保存LSE [B, H, N]
    # LSE_offsets = index_B * stride_B + index_H * stride_LSE_H + offsets_QO_N
    # = index_B * (H * stride_H) + index_H * stride_LSE_H + offsets_QO_N
    # =index_BH * stride_H + offsets_QO_N
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)
    
    # 2. 保存O [B, H, N, Dh]
    O_offsets = offsets_QO_N[:, None] * stride_O_N + offset_Dh[None, :] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])
    
@triton.autotune(
    configs=[
        triton.Config({'PRE_BLOCK_SIZE_ROW': PRE_BLOCK_SIZE_ROW}, num_stages=num_stages, num_warps=num_warps)
        for PRE_BLOCK_SIZE_ROW in [32] for num_stages in [3] for num_warps in [4]
    ],
    key=["Dh"]
)
@triton.jit
def attn_backward_preprocess(
    O_ptr, dLdO_ptr, Delta_ptr,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_dLdO_B, stride_dLdO_H, stride_dLdO_N, stride_dLdO_Dh,
    stride_Delta_B, stride_Delta_H, stride_Delta_N,
    N, Dh: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr,
):
    index_BH = tl.program_id(1)
    
    row = tl.program_id(0)
    
    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, Dh)
    
    mask = row_offsets < N
    
    # load O
    O_ptr += index_BH * stride_O_H
    O_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :] * stride_O_Dh
    O = tl.load(O_ptr + O_offsets, mask=mask[:, None], other=0.0) # shape [PRE_BLOCK_SIZE_ROW, Dh]
    
    # load dLdO
    dLdO_ptr += index_BH * stride_dLdO_H
    dLdO_offsets = row_offsets[:, None] * stride_dLdO_N + col_offsets[None, :] * stride_dLdO_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask[:, None], other=0.0)
    
    # 计算Delta
    # Delta = sum(dLdO * dO, dim=-1)
    Delta = tl.sum(dLdO.to(tl.float32) * O.to(tl.float32), axis=-1) # shape [PRE_BLOCK_SIZE_ROW]
    
    # save Delta
    Delta_ptr += index_BH * stride_Delta_H
    tl.store(Delta_ptr + row_offsets, Delta, mask=mask)

# 求解K和V的梯度，需要所有的Q所以，固定住K和V的大小，流动遍历Q
# K,V,dK,dV是固定在SRAM中的
@triton.jit
def _attn_backward_KV(
    K, V, dLdK, dLdV, # shape [BLOCK_SIZE_COL, Dh]
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr,
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    arrows indicate direction of this pid's for loop; each arrow is a different PID
    这个 kernel 负责计算特定一块 K 和 V 的梯度。
    它会遍历 Q 的序列长度 (rows) 来收集对当前 K/V 的贡献。
    """

    # 1. 初始化偏移
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)
    
    # 2. 计算Q和dLdO的指针偏移
    # 由于我们需要计算dST=K@QT，所以直接计算QT的偏移
    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh

    # 3. 遍历Q
    for block_idx in range(num_steps):
        # 4. 加载数据
        mask_N = offsets_ROW < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.) # [Dh, BLOCK_SIZE_ROW]
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.) # [BLOCK_SIZE_ROW, Dh]
        LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_N, other=0.) # [BLOCK_SIZE_ROW]
        Delta = tl.load(Delta_ptr + offsets_ROW, mask=mask_N, other=0.) # [BLOCK_SIZE_ROW)
        
        # 5. 计算S_T
        # [BLOCK_SIZE_COL, BLOCK_SIZR_ROW]
        S_T = tl.dot(K, Q_T)
        
        # 6. P_T
        # P_T = exp2(S_T - LSE), LSE = log2(sum(S))
        # [BLOCK_SIZE_COL, BLOCK_SIZR_ROW]
        P_T = tl.exp2(S_T - LSE[None, :])
        
        # 7. mask(对角线处)
        if MASK:
            # offsets_COL:[8, 9, 10, 11]
            # offsets_ROW:[8, 9, 10, 11]
            # 1, 1, 1, 1
            # 0, 1, 1, 1
            # 0, 0, 1, 1
            # 0, 0, 0, 1
            mask = offsets_COL[:, None] <= offsets_ROW[None, :]
            P_T = tl.where(mask, P_T, 0.0)
            
        # 8. dLdV
        # dLdV += P_T @ dO
        # P_T [BLOCK_SIZE_COL, BLOCK_SIZR_ROW], dLdO: [BLOCK_SIZE_ROW, Dh] -> [BLOCK_SIZE_COL, Dh]
        dLdV = tl.dot(P_T, dLdO, acc=dLdV) 
        
        # 9. dLdK
        # dLdK = dS_T @ Q
        # dS_T = P_T * (dP_T - Delta)
        # dP_T = V @ dO_T
        dP_T = tl.dot(V, tl.trans(dLdO)) # [BLOCK_SIZE_COL, Dh] @ [Dh, BLOCK_SIZE_ROW] -> [BLOCK_SIZE_COL, BLOCK_SIZR_ROW]
        dS_T = P_T * (dP_T - Delta) * ln2 # 前向时使用的是exp2，exp2求导=exp2 * ln2
        
        dLdK = tl.dot(dS_T, tl.trans(Q_T), acc=dLdK)
        
        # 10. 移动到下一个Q块
        offsets_ROW += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr += BLOCK_SIZE_ROW * stride_N
        
    return dLdK, dLdV
        
@triton.jit
def _attn_backward_Q(
    dLdQ,
    Q, dLdO, LSE, Delta,
    K_ptr, V_ptr,
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    
    start_ROW, start_COL,
    num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    # 1. 初始化偏移
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)
    
    # 2. 计算K和V的偏移量（转置）
    # shape: [Dh, BLOCK_SIZE_ROW]
    KV_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N
    
    # 3. 遍历K V
    for block_idx in range(num_steps):
        mask_COL = offsets_COL < N
        
        # 4. 加载K V
        # [Dh, BLOCK_SIZE_COL]
        K_T = tl.load(K_ptr + KV_T_offsets, mask=mask_COL[None, :], other=0.)
        V_T = tl.load(V_ptr + KV_T_offsets, mask=mask_COL[None, :], other=0.)
        
        # 5. 计算S
        # [row, d] @ [d, col] -> [row, col]
        S = tl.dot(Q, K_T)
        
        # 6. 计算P
        # [row, col]
        P = tl.exp2(S - LSE)
        
        if MASK:
            mask = offsets_ROW[:, None] >= offsets_COL[None, :]
            P = tl.where(mask, P, 0.0)
        
        # 7. 计算
        dLdP = tl.dot(dLdO, V_T)
        dLdS = P * (dLdP - Delta) * ln2
        dLdQ += tl.dot(dLdS, tl.trans(K_T))
        
        # 8. 移动指针
        offsets_COL += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_N
        V_ptr += BLOCK_SIZE_COL * stride_N

    return dLdQ

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_MICRO in [16]#, 32, 64]
        for BLOCK_SIZE_MACRO in [32]#, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
        if BLOCK_SIZE_MACRO > BLOCK_SIZE_MICRO # could do >= but i wanna get mileage out of the loop code we wrote
    ],
    key=["Dh"],
)
@triton.jit
def attn_backward(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr,
    dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr,
):
    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634

    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // H
    index_head = index_batch_head % H
    
    batch_head_offset = index_batch * stride_B + index_head * stride_H
    Q_ptr += batch_head_offset
    K_ptr += batch_head_offset
    V_ptr += batch_head_offset
    dLdO_ptr += batch_head_offset
    dLdQ_ptr += batch_head_offset
    dLdK_ptr += batch_head_offset
    dLdV_ptr += batch_head_offset
    
    # LSE 和 Delta: [B, H, N] stride_H = N
    batch_head_offset_LSE = index_batch_head * N
    LSE_ptr += batch_head_offset_LSE
    Delta_ptr += batch_head_offset_LSE
    
    # =========================================================
    # Phase 1: 计算 dK 和 dV
    # 策略：固定 K/V (MACRO), 遍历 Q (MICRO)
    # =========================================================
    pid = tl.program_id(0)
    
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO # 固定的KV宽度
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO # 流动的Q宽度
    
    # KV的起点
    start_COL = pid * BLOCK_SIZE_COL_1
    # Q的起点必须要大于等于KV的起点，因为是causal mask
    start_ROW = start_COL
    # 对角线区域
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1
    # N_adj = tl.cdiv(N, BLOCK_SIZE_ROW_1) * BLOCK_SIZE_ROW_1
    # num_steps = (N_adj - start_ROW) // BLOCK_SIZE_ROW_1
    
    # 加载固定的K V
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    kv_loc = offsets_COL[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    kv_mask = offsets_COL < N
    
    # shape: [COL, Dh]
    K = tl.load(K_ptr + kv_loc, mask=kv_mask[:, None], other=0.)
    V = tl.load(V_ptr + kv_loc, mask=kv_mask[:, None], other=0.)
    
    # 初始化dLdK和dLdV
    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    
    K *= scale * rln2
    
    # 调用Inner Kernel
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr,
        LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )
    # 非对角线区域，不需要MASK
    start_ROW += BLOCK_SIZE_COL_1
    N_adj = tl.cdiv(N, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1
    num_steps = (N_adj - start_ROW) // BLOCK_SIZE_ROW_1
    
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr,
        LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )
    dLdK *= scale * rln2
    tl.store(dLdK_ptr + kv_loc, dLdK, mask=kv_mask[:, None])
    tl.store(dLdV_ptr + kv_loc, dLdV, mask=kv_mask[:, None])

    # =========================================================
    # Phase 2: 计算 dQ
    # 策略：固定 Q (MACRO), 遍历 K/V (MICRO)
    # =========================================================
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO
    
    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    
    # 对角线区域
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    q_loc = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    q_mask = offsets_ROW < N

    Q = tl.load(Q_ptr + q_loc, mask=q_mask[:, None], other=0.)
    dLdO = tl.load(dLdO_ptr + q_loc, mask=q_mask[:, None], other=0.)
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=q_mask, other=0.)[:, None]
    Delta = tl.load(Delta_ptr + offsets_ROW, mask=q_mask, other=0.)[:, None]
    
    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)
    Q *= scale * rln2

    # 调用Inner Kernel 对角线区域，需要MASK
    dLdQ = _attn_backward_Q(
        dLdQ,
        Q, dLdO, LSE, Delta,
        K_ptr, V_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL,
        num_steps,
        scale, ln2, rln2,
        MASK=True
    )
    
    end_COL = start_COL
    start_COL = 0
    num_steps = end_COL // BLOCK_SIZE_COL_2
    
    # 非对角线区域，不需要MASK
    dLdQ = _attn_backward_Q(
        dLdQ,
        Q, dLdO, LSE, Delta,
        K_ptr, V_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL,
        num_steps,
        scale, ln2, rln2,
        MASK=False
    )
    
    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + q_loc, dLdQ, mask=q_mask[:, None])
    
    
    
class _flash_attetion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        # qkv shape: [batch_size, num_heads, seq_len, head_dim]
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        B, H, N, Dh = q.shape
        assert q.shape == k.shape == v.shape
        assert Dh <=128, "Flash Attention only supports head_dim <= 128"
        assert q.dtype == k.dtype == v.dtype == torch.float32, "Flash Attention only supports float32"
        
        # 预分配输出显存
        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
        
        # 定义 Grid
        # forward pass 主要按 Q 的分块 (BLOCK_SIZE_QO) 并行
        grid = lambda META: (
            triton.cdiv(N, META['BLOCK_SIZE_QO']),
            B * H
        )
        
        # 调用前向传播 kernel
        attn_fwd[grid](
            q, k, v,
            O,
            LSE,
            scale,
            # strides
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            # sizes
            B, H, N, Dh,
        )
        
        # 保存 ctx 以便 backward 使用
        # save_for_backward 会将 Tensor 存入显存（如果是推理模式则不存）
        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid # 保存 grid 计算逻辑（或直接重新定义）
        ctx.scale = scale
        ctx.B = B
        ctx.H = H
        ctx.N = N
        ctx.Dh = Dh
        
        return O
        
    
    @staticmethod
    def backward(ctx, dLdO):
        """
        反向传播接口
        输入: dLdO (Loss 对 O 的梯度)
        输出: dLdq, dLdk, dLdv, None (对应 forward 的 4 个输入)
        """
        q, k, v, O, LSE = ctx.saved_tensors
        scale = ctx.scale
        B = ctx.B
        H = ctx.H
        N = ctx.N
        Dh = ctx.Dh
        
        dLdO = dLdO.contiguous()
        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)
        
        Delta = torch.empty_like(LSE)
        
        # ====================================================
        # Step A: 启动 Preprocess Kernel (计算 Delta)
        # ====================================================
        pre_grid = lambda META:(
            triton.cdiv(N, META['PRE_BLOCK_SIZE_ROW']),
            B * H
        )
        
        attn_backward_preprocess[pre_grid](
            O, dLdO, Delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dLdO.stride(0), dLdO.stride(1), dLdO.stride(2), dLdO.stride(3),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            N, Dh,
        )
        # ====================================================
        # Step B: 启动 Main Backward Kernel (计算 dQ, dK, dV)
        # ====================================================
        # 注意：这里我们基于 MACRO 块大小来划分 Grid
        grid = lambda META:(
            triton.cdiv(N, META['BLOCK_SIZE_MACRO']),
            B * H
        )
        attn_backward[grid](
            q, k, v,
            dLdO,
            dLdq, dLdk, dLdv,
            LSE, Delta,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, N, Dh,
        )
        # 4. 返回梯度
        # 顺序必须严格对应 forward 的参数顺序: q, k, v, scale
        # scale 是标量，没有梯度，返回 None
        return dLdq, dLdk, dLdv, None

triton_flash_attention = _flash_attetion.apply
        
    
def test_flash_attention_kernel(B, H, N, Dh, device=DEVICE, atol=5e-3):
    q = torch.randn(B, H, N, Dh, dtype=torch.float32, device=DEVICE, requires_grad=True)
    k = torch.randn(B, H, N, Dh, dtype=torch.float32, device=DEVICE, requires_grad=True)
    v = torch.randn(B, H, N, Dh, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scale = 1 / math.sqrt(Dh)

    tri_out = triton_flash_attention(q, k, v, scale)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    triton.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
    print("-> Passed Forward")
    
    # Triton 反向传播
    dLdO = torch.randn_like(q) * 0.1
    tri_out.backward(dLdO, retain_graph=True)
    
    dLdq_tri, dLdk_tri, dLdv_tri = [_.grad.clone() for _ in (q, k, v)]
    
    q.grad, k.grad, v.grad = None, None, None
    
    # PyTorch 反向传播
    ref_out.backward(dLdO, retain_graph=True)
    dLdq_ref, dLdk_ref, dLdv_ref = [_.grad.clone() for _ in (q, k, v)]
    q.grad, k.grad, v.grad = None, None, None
    
    # test
    torch.testing.assert_close(dLdq_tri, dLdq_ref, atol=atol, rtol=0)
    torch.testing.assert_close(dLdk_tri, dLdk_ref, atol=atol, rtol=0)
    torch.testing.assert_close(dLdv_tri, dLdv_ref, atol=atol, rtol=0)
    print("-> Passed Backward")
    
configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"], # 横轴，序列长度
            x_vals=[512 * i for i in range(1, 17)], # 序列长度从512到8192
            line_arg="provider", # 不同的provider画不同的线
            line_vals=["triton", "pytorch"], # triton和pytorch两种实现
            line_names=[
                "Triton Flash Attention",
                "PyTorch Scaled Dot-Product Attention"
            ], # 线的名字
            styles=[("green", "-"), ("blue", "-")], # 线的风格
            ylabel="TFLOPS", # 纵轴: TFLOPS
            plot_name=f"Flash Attention {mode} pass performance", # 图名字
            args={"mode": mode} # 传给benchmark的额外参数
        )
    )
@triton.testing.perf_report(configs)
def benchmark_flash_attention(SEQ_LEN, provider, mode, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    # 固定batch，head_dim，只变化seq_len
    BATCH, N_HEADS = 32, 4
    HEAD_DIM = 128
    
    q = torch.randn([BATCH, N_HEADS, SEQ_LEN, HEAD_DIM], dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn([BATCH, N_HEADS, SEQ_LEN, HEAD_DIM], dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn([BATCH, N_HEADS, SEQ_LEN, HEAD_DIM], dtype=torch.float32, device=device, requires_grad=True)
    scale = 1 / math.sqrt(HEAD_DIM)
    
    if provider == "triton":
        fn = lambda: triton_flash_attention(q, k, v, scale)
    if provider == "pytorch":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
        
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    
    flops_per_matmul = 2 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    
    total_flops = 2 * flops_per_matmul # 前向传播有2次matmul
    
    if mode == "bwd":
        total_flops *= 2.5
        
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    # 单元测试
    test_flash_attention_kernel(1, 1, 128, 32) # without block masking
    test_flash_attention_kernel(1, 1, 128, 64) # without block masking
    test_flash_attention_kernel(1, 1, 128, 128) # without block masking
    test_flash_attention_kernel(32, 8, 69, 128) # with block masking
    
    import sys
    save_path = '/content/drive/MyDrive'
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_flash_attention.run(save_path=save_path, print_data=True)

    
!ls -lh

# 核心思想
FlashAttention 的核心在于减少 GPU HBM (显存) 的读写次数。标准 Attention 需要计算并存储 $N \times N$ 的巨大矩阵，而 FlashAttention 通过分块计算，将计算过程限制在快速的 SRAM 中。
# 前向传播
已知量：Q, K, V -> [N, D];

待求量：O -> [N, D]; M, LSE -> [N,]

计算公式：

$O = P * V$ -> [N, D] = [N, N] * [N, D]

$P = Softmax(S)$;  

$S = \frac{Q * K^T}{\sqrt{D}}$

Online Softmax:

$P = \frac{exp(S-M)}{\sum{exp{S-M}}}$


基本计算流 (The "Fixed Q, Flowing K/V" Strategy):

并行维度：Grid 主要按 $N$ 维度（Sequence Length）划分。每个 Program ID (PID) 负责计算一个 BLOCK_SIZE_QO 大小的 Q 块。

Triton中的grid有两个维度，第一个维度为Q的BLOCK ID，第二个维度为Head ID：

```python
grid = lambda META: (
    triton.cdiv(N, META['BLOCK_SIZE_QO']),
    B * H
)
```

驻留 (Resident)：当前 PID 加载对应的 Q 块 到 SRAM 中。在整个 Inner Loop 过程中，这个 Q 块保持不变。

流动 (Scanning)：K 块 和 V 块 从 HBM 中被分批加载到 SRAM（滑动窗口）。

计算：Q 块与当前的 K 块计算分数，更新 Softmax，与 V 块相乘，累加到输出 O 中，然后丢弃当前的 K/V，加载下一块。

关于MASK：Q只能看到它和它之前的K，对于小于Q的K来说，不需要MASK，对于和Q有重叠的K来说，这个单独的BLOCK里需要MASK。

比如现在的Q BLOCK是Q的64~127行，那么0~63的K不需要MASK,64~127的K需要MASK。

具体实现是根据是否需要MASK，来改变K和V的下界和上界。比如我现在需要计算64~127的Q块，如果需要MASK，那么就只加载[64, 127]的K和V，并且进行mask处理：

```python
causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
S += tl.where(causal_mask, 0, -1.0e6)
```

如果不需要MASK，那么就只加载，[0, 63]的K和V，这时直接计算，不需要mask处理，至于128及以上的，直接不加载。

具体流程：

根据是否需要mask，确定K和V的加载上下界。然后遍历界内：

1. 固定在SRAM中的是$Q_i%，现在遍历到的是$K_j$和$V_j$，计算$S_{ij} = Q_i * K_j^T$ # shape: [BLOCK_SIZE_QO, BLOCK_SIZE_KV]

2. 更新全局最大值，$M_{new} = max(max(S_{ij}), M_{old})$ # shape: [BLOCK_SIZE_QO]

3. 计算修正因子, $\alpha=exp(M_{old} - M_{new})$

4. 计算P，$P = exp(S - M_{new})$

5. 更新分母和LSE, $L_{new} = sum(P, axis=1)$, $LSE = LSE * \alpha + L_{new}$

6. 更新输出O， $O = O * \alpha + P * V$

7. 更新全局最大值，$M = M_{new}$

8. 移动K和V的指针

代码：

```python
# 遍历K和V的分块
    for start_kv in range(low, high, BLOCK_SIZE_KV):
        # 编译器优化提示
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # KV的行的mask
        mask_KV_N = offsets_KV_N < N

        # 加载K_T，注意因为是转置，所以mask也要转置，mask_KV_N[None, :]
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)

        # Q和K_T相乘
        S = tl.dot(Q, K_T) * scale
        
        if DIAGONAL:
            # offsets_QO_N:[8, 9, 10, 11]
            # offsets_KV_N:[8, 9, 10, 11]
            # 0, 1, 1, 1
            # 0, 0, 1, 1
            # 0, 0, 0, 1
            # 0, 0, 0, 0 
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            S += tl.where(causal_mask, 0, -1.0e6)
        
        # Online Softmax
        # # 需要不断更新每一行的Max值
        # # [BLOCK_SIZE_QO]
        # m_cur_block = tl.max(S, axis=1)

        # # 更新全局最大值
        # # [BLOCK_SIZE_QO]
        # M_new = tl.maximum(m_cur_block, M)
        
        M_new = tl.maximum(M, tl.max(S, axis=1))
        
        # 修正S并计算P
        # P = exp(S - M_new)
        S -= M_new[:, None]
        P = tl.exp2(S)
        
        # 更新分母和L
        # [BLOCK_SIZE_QO]
        L_new = tl.sum(P, axis=1)
        # 根据新的max值，计算修正系数
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new
        
        # 更新O的输出
        # O_new = O_old * alpha + P @ V
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
        O = O * alpha[:, None]
        O = tl.dot(P, V, acc=O)
        
        # 更新M [BLOCK_SIZE_QO]
        M = M_new
        
        # 移动指针到下一个Block
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
```

# 后向传播

## 公式推导

前向传播的公式为：

$S = scale * (Q * K^T)$

$P = softmax(S) = \frac{exp(S-M)}{\sum{exp(S-M)}}$

$O = P * V$

1. 推导$dV$

$dV = P^T * dO$

2. 推导$dS$

$dP = dO * V^T$

$$
对于一行S=\{s_1, s_2, s_3, s_N\},S_i的梯度dS_{i} = \sum_{j}dP_j * \frac{\partial P_j}{\partial S_i} \\


当i=j时，\frac{\partial P_j}{\partial S_i} = P_j * (1-P_j) \\
当i\neq j时，\frac{\partial P_j}{\partial S_i}  = -P_j*P_i \\
所以，dS_i = \sum_{j} dP_j * (-P_j*P_i)+dP_i*P_i*(1-P_i)=\sum_{j} dP_j * (-P_j*P_i) + dP_i * dP_j \\
= P_i * (dP_i-\sum_{i} dP_j * dP_j)
$$
接下来求$\sum_{i} dP_j * dP_j$

$$
对于第i行的Q，\Delta _{i}=\sum_{k} dP_{ik} * dP_{ik} \\
dP_{ik} = dO_i * V_k^T = \sum _{d} dO_{id} * V_{dk}^T = \sum _{d} dO_{id} * V_{kd} \\
\therefore \Delta _{i}=\sum_{k} P_{ik} \sum_{d}dO_{id}V_{kd}=\sum_{d}dO_{id}\sum_{k}P_{ik}V_{kd} (求和的交换律) \\
=\sum_{d}dO_{id}*O_{id}
\therefore \Delta_{i} = sum(dO * o, axis=-1) \\
\therefore dS = P \circ (dP - \Delta), 其中 \circ 表示逐元素乘法 (Element-wise multiplication)。
$$

3. 推导$dQ, dK$

$$
根据公式S = scale * (Q * K^T)可求得\\
dQ = dS \cdot K \cdot scale\\
dK = dS^T \cdot Q \cdot scale

$$



## Triton实现流程

1. 预处理 (Preprocessing Kernel)

+ 目的：计算辅助变量 $\Delta$。

+ 输入：$O, dO$

+ 操作：Delta = sum(O * dO, axis=-1)

+ 输出：$\Delta$ (Shape: [B, H, N])
2. 反向传播的主逻辑

### 固定Q，流动K和V

此时grid按照K进行分块（BLOCK_SIZE_KV），每个kernel处理一个BLOCK的K和V，把该BLOCK加载到SRAM中，流动Q、dO、LSE、Delta。

首先要重新计算$S = QK^T$ [BLOCK_SIZE_Q, D] * [D, BLOCK_SIZE_KV] = [BLOCK_SIZE_Q, BLOCK_SIZE_KV]

然后计算$P=exp(S-M)$

$dV += P^T \cdot dO$   

shape: [BLOCK_SIZE_KV, BLOCK_SIZE_Q] * [BLOCK_SIZE_Q, D] = [BLOCK_SIZE_KV, D]

$dK += dS^T \cdot Q (dS = P (dP - \Delta))$ 

shape:  [BLOCK_SIZE_KV, BLOCK_SIZE_Q] * [BLOCK_SIZE_Q, D] = [BLOCK_SIZE_KV, D]
### 固定Q，流动K和V

此时grid按照Q进行分块（BLOCK_SIZE_Q），每个kernel处理一个BLOCK的Q，把该BLOCK的Q以及dO,LSE,Delta加载到SRAM中，流动K、V.

重计算P

累积梯度： $dQ += dS \cdot K$


关于固定和流动，以及梯度累加，可以看下图：

<img src="./1.png" alt="Example Image" style="background-color:white;" />


