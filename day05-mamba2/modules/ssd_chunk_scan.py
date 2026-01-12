import math
from packaging import version

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")

# Step 5: Calculate C * states * dA_cumsm + CB * dA_cumsum * dt * x + D * x
"""
Input:
- cb: [batch, nchunks, ngroups, chunk_size, chunk_size]
- x: [batch, seq_len, nheads, head_dim]
- dt: [batch, nheads, nchunks, chunk_size]
- dA_cumsum: [batch, nheads, nchunks, chunk_size]
- C: [batch, seq_len, ngroups, d_state]
- states: [batch, nchunks, nheads, head_dim, d_state]
- D: [nheads, head_dim] or [nheads]
- z: [batch, seq_len, nheads, head_dim]

Mathematical Formula:
$$ Y = C \cdot states \cdot dA_cs + \sum_{t=0}^{T-1} CB \cdot L \cdot X \cdot dt $$

Output:
- out: [batch, seq_len, nheads, head_dim]
- out_x: [batch, seq_len, nheads, head_dim] if z is not None else None
"""
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'head_dim', 'd_state', 'IS_CAUSAL'],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    C_ptr, prev_states_ptr, D_ptr,
    chunk_size, head_dim, d_state,
    batch, seq_len, nheads_ngroups_ratio,
    stride_cb_batch, stride_cb_nchunks, stride_cb_ngroups, stride_cb_chunk_size_m, stride_cb_chunk_size_k,
    stride_x_batch, stride_x_seq_len, stride_x_nheads, stride_x_head_dim,
    stride_z_batch, stride_z_seq_len, stride_z_nheads, stride_z_head_dim,
    stride_out_batch, stride_out_seq_len, stride_out_nheads, stride_out_head_dim,
    stride_dt_batch, stride_dt_nchunks, stride_dt_nheads, stride_dt_chunk_size,
    stride_dA_cs_batch, stride_dA_cs_nchunks, stride_dA_cs_nheads, stride_dA_cs_chunk_size,
    stride_seq_idx_batch, stride_seq_idx_seq_len,
    stride_C_batch, stride_C_seq_len, stride_C_ngroups, stride_C_d_state,
    stride_states_batch, stride_states_nchunks, stride_states_nheads, stride_states_head_dim, stride_states_d_state,
    stride_D_nheads,
    # Meta-parameter
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(1)
    pid_chunk = pid_bc // batch
    pid_batch = pid_bc - pid_chunk * batch
    pid_head = tl.program_id(2)
    num_pid_n = tl.cdiv(d_state, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n
    
    cb_ptr += pid_batch * stride_cb_batch + pid_chunk * stride_cb_nchunks + (pid_head // nheads_ngroups_ratio) * stride_cb_ngroups
    x_ptr += pid_batch * stride_x_batch + pid_chunk * chunk_size * stride_x_seq_len + pid_head * stride_x_nheads
    dt_ptr += pid_batch * stride_dt_batch + pid_chunk * stride_dt_nchunks + pid_head * stride_dt_nheads
    dA_cumsum_ptr += pid_batch * stride_dA_cs_batch + pid_chunk * stride_dA_cs_nchunks + pid_head * stride_dA_cs_nheads
    C_ptr += pid_batch * stride_C_batch + pid_chunk * chunk_size * stride_C_seq_len + (pid_head // nheads_ngroups_ratio) * stride_C_ngroups
    prev_states_ptr += pid_batch * stride_states_batch + pid_chunk * stride_states_nchunks + pid_head * stride_states_nheads
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_batch * stride_seq_idx_batch + pid_chunk * chunk_size * stride_seq_idx_seq_len

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # [BLOCK_SIZE_M]
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_chunk_size, mask=offs_m < chunk_size, other=0.).to(tl.float32)
    
    chunk_size_limit = min(chunk_size, seq_len - pid_chunk * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seq_len, mask=pid_chunk >= 1, other=-1)
        # [BLOCK_SIZE_M]
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seq_len, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    ####### Step 1: Calculate $$  C \cdot states \cdot dA_cs $$ #######
    if IS_TRITON_22 or pid_chunk > -1:
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        # C: [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
        C_ptrs = C_ptr + offs_m[:, None] * stride_C_seq_len + offs_k_dstate[None, :] * stride_C_d_state
        # prev_states: [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N]
        prev_states_ptrs = prev_states_ptr + offs_k_dstate[:, None] * stride_states_d_state + offs_n[None, :] * stride_states_head_dim
        if not HAS_SEQ_IDX:
            # [BLOCK_SIZE_M]
            scale_m = tl.exp(dA_cs_m)
        else:
            # [BLOCK_SIZE_M]
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            # C: [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < d_state), other=0.)
            # prev_states: [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N]
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < d_state) & (offs_n[None, :] < head_dim), other=0.)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            # C: [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] * [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, d_state, BLOCK_SIZE_K):
                # C: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < d_state - k), other=0.)
                # # prev_states: [BLOCK_SIZE_K, BLOCK_SIZE_N]
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < d_state - k) & (offs_n[None, :] < head_dim), other=0.)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                # [BLOCK_SIZE_M, BLOCK_SIZE_K] * [BLOCK_SIZE_K, BLOCK_SIZE_N] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K * stride_C_d_state
                prev_states_ptrs += BLOCK_SIZE_K * stride_states_d_state
            acc *= scale_m[:, None]
            
    ####### Step 2: Calculate $$ \sum_{t=0}^{T-1} CB \cdot L \cdot X \cdot dt $$  #######
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # cb: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    cb_ptrs = cb_ptr + offs_m[:, None] * stride_cb_chunk_size_m + offs_k[None, :] * stride_cb_chunk_size_k
    # x: [BLOCK_SIZE_K, BLOCK_SIZE_N]
    x_ptrs = x_ptr + offs_k[:, None] * stride_seq_idx_seq_len + offs_n[None, :] * stride_x_head_dim
    # dt: [BLOCK_SIZE_K]
    dt_ptrs = dt_ptr + offs_k * stride_dt_chunk_size
    # dA_cumsum: [BLOCK_SIZE_K]
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_chunk_size
    # If causal, the index of col is <= row. 
    # For example, the index of col is [8, 9, 10, 11], the offs_k must be <= 11.
    # The start of offs_k is [0, 1, 2, 3], then the end of loop is 12 == (2 + 1) * 4
    end_of_loop = chunk_size_limit if not IS_CAUSAL else min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    for k in range(0, end_of_loop, BLOCK_SIZE_K):
        # cb: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.).to(tl.float32)
        # dA_cs_k: [BLOCK_SIZE_K]
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.).to(tl.float32)
        # decay: $$ A_{m:k} = $$ dA_cs_m - dA_cs_k [BLOCK_SIZE_M, BLOCK_SIZE_K]
        cb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_k[None, :]), 0.0))
        # dt_k: [BLOCK_SIZE_K]
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.).to(tl.float32)
        cb *= dt_k[None, :]
        if IS_CAUSAL:
            mask = offs_m[:, None] >= offs_k[None, :] + k
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        # x: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < head_dim), other=0.)
        # [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] * [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_chunk_size_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seq_len
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_chunk_size
        dt_ptrs += BLOCK_SIZE_K * stride_dt_chunk_size
        
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    if HAS_D:
        if D_HAS_HDIM: # D: [nheads, head_dim]
            # D: [1, BLOCK_SIZE_N]
            D = tl.load(D_ptr + pid_head * stride_D_nheads + offs_n * 1, mask=offs_n < head_dim, other=0.).to(tl.float32)
        else:
            # D: [1]
            D = tl.load(D_ptr + pid_head * stride_D_nheads).to(tl.float32)
            # x_residual: [BLOCK_SIZE_M, BLOCK_SIZE_N]
            x_residual = tl.load(x_ptr + offs_m[:, None] * stride_x_seq_len + offs_n[None, :] * stride_x_head_dim,
                                 mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim), other=0.).to(tl.float32)
            acc += x_residual * D
    if HAS_Z:
        out_x_ptr += pid_batch * stride_out_batch + pid_head * stride_out_nheads + pid_chunk * chunk_size * stride_out_seq_len
        out_x_ptrs = out_x_ptr + offs_out_m[:, None] * stride_out_seq_len + offs_out_n[None, :] * stride_out_head_dim
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < head_dim))
        
        z_ptr += pid_batch * stride_z_batch + pid_head * stride_z_nheads + pid_chunk * chunk_size * stride_z_seq_len
        z_ptrs = z_ptr + offs_out_m[:, None] * stride_z_seq_len + offs_out_n[None, :] * stride_z_head_dim
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < head_dim), other=0.).to(tl.float32)
        acc *= z * tl.sigmoid(z)
        
    out_ptr += pid_batch * stride_out_batch + pid_head * stride_out_nheads + pid_chunk * chunk_size * stride_out_seq_len
    out_ptrs = out_ptr + offs_out_m[:, None] * stride_out_seq_len + offs_out_n[None, :] * stride_out_head_dim
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < head_dim))
    
def _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=None, z=None, seq_idx=None):
    batch, seq_len, nheads, head_dim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, d_state = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seq_len, ngroups, d_state)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, head_dim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, head_dim, d_state)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    
    out = torch.empty((batch, seq_len, nheads, head_dim), device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty((batch, seq_len, nheads, head_dim), device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(head_dim, META['BLOCK_SIZE_N']),
                         batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0))
    with torch.cuda.device(device=x.device.index):
        _chunk_scan_fwd_kernel[grid](
            cb, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, states, D,
            chunk_size, head_dim, d_state,
            batch, seq_len, nheads // ngroups,
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            D.stride(0) if D is not None else 0,
            True,
            D is not None,
            D.dim() == 2 if D is not None else True,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(d_state), 16),
            HAS_Z=z is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_TRITON_22=TRITON_22,
        )
        return out, out_x