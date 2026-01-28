import math
from packaging import version

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.utils.determinism import (
    alloc_tile_workspace,
    finalize_tile_workspace,
    use_deterministic_mode,
    autotune_configs,
)
def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

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
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 64}),
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
    ],
    key=['chunk_size', 'head_dim'],
)
@triton.jit
def _chunk_scan_bwd_dz_kernel(
    dout_ptr, out_ptr, z_ptr, x_ptr, D_ptr, outz_ptr, dz_ptr, dout_x_ptr, dD_ptr, ddA_cumsum_ptr,
    chunk_size, head_dim,
    batch, seq_len,
    # strides
    stride_dout_batch, stride_dout_seq_len, stride_dout_nheads, stride_dout_head_dim,
    stride_out_batch, stride_out_seq_len, stride_out_nheads, stride_out_head_dim,
    stride_z_batch, stride_z_seq_len, stride_z_nheads, stride_z_head_dim,
    stride_x_batch, stride_x_seq_len, stride_x_nheads, stride_x_head_dim,
    stride_D_nheads,
    stride_outz_batch, stride_outz_seq_len, stride_outz_nheads, stride_outz_head_dim,
    stride_dz_batch, stride_dz_seq_len, stride_dz_nheads, stride_dz_head_dim,
    stride_dout_x_batch, stride_dout_x_seq_len, stride_dout_x_nheads, stride_dout_x_head_dim,
    stride_dD_batch, stride_dD_chunk, stride_dD_nheads, stride_dD_csize, stride_dD_head_dim,
    stride_ddA_cumsum_batch, stride_ddA_cumsum_chunk, stride_ddA_cumsum_nheads, stride_ddA_cumsum_csize,
    # Meta-parameter
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_DDACSM: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_batch_chunk = tl.program_id(1)
    pid_chunk = pid_batch_chunk // batch
    pid_batch = pid_batch_chunk - pid_chunk * batch
    pid_head = tl.program_id(2)
    pid_m = tl.program_id(0)
    
    # Move the pointers to [batch, chunk, head]
    dout_ptr += pid_batch * stride_dout_batch + pid_chunk * chunk_size * stride_dout_seq_len + pid_head * stride_dout_nheads
    dout_x_ptr += pid_batch * stride_dout_x_batch + pid_chunk * chunk_size * stride_dout_x_seq_len + pid_head * stride_dout_x_nheads
    out_ptr += pid_batch * stride_out_batch + pid_chunk * chunk_size * stride_out_seq_len + pid_head * stride_out_nheads
    z_ptr += pid_batch * stride_z_batch + pid_chunk * chunk_size * stride_z_seq_len + pid_head * stride_z_nheads
    dz_ptr += pid_batch * stride_dz_batch + pid_chunk * chunk_size * stride_dz_seq_len + pid_head * stride_dz_nheads
    if RECOMPUTE_OUTPUT:
        outz_ptr += pid_batch * stride_outz_batch + pid_chunk * chunk_size * stride_outz_seq_len + pid_head * stride_outz_nheads
    if HAS_DDACSM:
        ddA_cumsum_ptr += pid_batch * stride_ddA_cumsum_batch + pid_chunk * stride_ddA_cumsum_chunk + pid_head * stride_ddA_cumsum_nheads
    if HAS_D:
        x_ptr += pid_batch * stride_x_batch + pid_chunk * chunk_size * stride_x_seq_len + pid_head * stride_x_nheads
        # Move to [batch, chunk, head, csize_block]
        dD_ptr += pid_batch * stride_dD_batch + pid_chunk * stride_dD_chunk + pid_head * stride_dD_nheads + pid_m * stride_dD_csize
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + offs_m[:, None] * stride_dout_seq_len + offs_n[None, :] * stride_dout_head_dim
    dout_x_ptrs = dout_x_ptr + offs_m[:, None] * stride_dout_x_seq_len + offs_n[None, :] * stride_dout_x_head_dim
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_seq_len + offs_n[None, :] * stride_out_head_dim
    z_ptrs = z_ptr + offs_m[:, None] * stride_z_seq_len + offs_n[None, :] * stride_z_head_dim
    dz_ptrs = dz_ptr + offs_m[:, None] * stride_dz_seq_len + offs_n[None, :] * stride_dz_head_dim
    if RECOMPUTE_OUTPUT:
        outz_ptrs = outz_ptr + offs_m[:, None] * stride_outz_seq_len + offs_n[None, :] * stride_outz_head_dim
    if HAS_D:
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_seq_len + offs_n[None, :] * stride_x_head_dim
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_head_dim
    
    chunk_size_limit = min(chunk_size, seq_len - pid_chunk * chunk_size)
    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim), other=0.).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim), other=0.).to(tl.float32)
    z = tl.load(z_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim), other=0.).to(tl.float32)
    z_sigmoid = tl.sigmoid(z)
    if RECOMPUTE_OUTPUT:
        # out = SSM(x) * SILU(z) = SSM(x) * z * sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        outz = out * z * z_sigmoid
        tl.store(outz_ptrs, outz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim))
    # dz = dO * SSM(x) * 
    dz = dout * out * z_sigmoid * (1 + z * (1 -z_sigmoid))
    tl.store(dz_ptrs, dz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim))
    # dSSM(x) = dO * SILU(z)
    dout *= z * z_sigmoid
    tl.store(dout_x_ptrs, dout, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim))
    if HAS_D:
        # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < head_dim), other=0.).to(tl.float32)
        if D_HAS_HDIM:
            # dD = sum(dO * x, axis=0)
            # [BLOCK_SIZE_N]
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < head_dim)
            # D: [BLOCK_SIZE_N]
            D = tl.load(D_ptr + pid_head * stride_D_nheads + offs_n, mask=offs_n < head_dim, other=0.).to(tl.float32)
        else:
            # [1]
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_head * stride_D_nheads).to(tl.float32)
        out -= x * D
    if HAS_DDACSM:
        # ddA_cs = sum(dO * SSM(x), axis=1) because ddA_cs's shape is [batch, seq_len, nheads], we need reduction of head_dim.
        ddA_cs = tl.sum(dout * out, axis=1)
        tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cumsum_csize, ddA_cs, mask=offs_m < chunk_size)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_scan_bwd_dstates_kernel(
    dout_ptr, C_ptr, dprev_states_ptr, dA_cumsum_ptr, seq_idx_ptr,
    head_dim, d_state, chunk_size,
    batch, seq_len, nchunks, nheads_ngroups_ratio,
    # strides
    stride_dout_batch, stride_dout_seq_len, stride_dout_nheads, stride_dout_head_dim,
    stride_C_batch, stride_C_seq_len, stride_C_ngroups, stride_C_d_state,
    stride_dprev_states_batch, stride_dprev_states_chunk, stride_dprev_states_nheads, stride_dprev_states_head_dim, stride_dprev_states_d_state,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_nheads, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seq_len,
    # Meta-parameter
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_bc = tl.program_id(2)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(1)
    num_pid_n = tl.cdiv(d_state, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n

    # Move pointers
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seq_len + (pid_h // nheads_ngroups_ratio) * stride_C_ngroups
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seq_len + pid_h * stride_dout_nheads
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_nheads
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seq_len
    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # [head_dim, chunk_size] -> [BLOCK_SIZE_M, BLOCK_SIZE_K]
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_head_dim + offs_k[None, :] * stride_dout_seq_len)
    # [chunk_size, d_state] -> [BLOCK_SIZE_K, BLOCK_SIZE_N]
    C_ptrs = C_ptr + (offs_k[:, None] * stride_C_seq_len + offs_n[None, :] * stride_C_d_state)
    # [chunk_size] -> [BLOCK_SIZE_K]
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seq_len
    chunk_size_limit = min(chunk_size, seq_len - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - 1 * stride_seq_idx_seq_len, mask=pid_c >= 1, other=0)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < head_dim) & (offs_k[None, :] < chunk_size_limit - k), other=0.).to(tl.float32)
        C = tl.load(C_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < d_state), other=0.).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale_k = tl.exp(dA_cs_k)
        else:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=0.).to(tl.float32)
            scale_k = tl.where(seq_idx_k==seq_idx_prev, tl.exp(dA_cs_k), 0.0)
        # [M, K] * [1, K] -> [M, K]
        dout = (dout * scale_k[None, :]).to(dout_ptr.dtype.element_ty)
        # [M, K] @ [K, N] -> [M, N]
        acc += tl.dot(dout, C)
        # Move pointers to next K block
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seq_len
        C_ptrs += BLOCK_SIZE_K * stride_C_seq_len
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seq_len
    out = acc.to(dprev_states_ptr.dtype.element_ty)
    dprev_states_ptr += pid_b * stride_dprev_states_batch + pid_c * stride_dprev_states_chunk + pid_h * stride_dprev_states_nheads
    dprev_states_ptrs = dprev_states_ptr + offs_m[:, None] * stride_dprev_states_head_dim + offs_n[None, :] * stride_dprev_states_d_state
    tl.store(dprev_states_ptrs, out, mask=(offs_m[:, None] < head_dim) & (offs_n[None, :] < d_state))

def _chunk_scan_bwd_dz(x, z, out, dout, chunk_size, has_ddAcs=False, D=None, dz=None, recompute_output=False):
    batch, seq_len, nheads, head_dim = x.shape
    assert z.shape == x.shape
    assert out.shape == x.shape
    assert dout.shape == out.shape
    nchunks = math.ceil(seq_len / chunk_size)
    if D is not None:
        assert D.shape == (nheads, head_dim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
    if has_ddAcs:
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
    if D is not None:
        BLOCK_SIZE_min = 32
        # [chunk_size // 32, batch, nchunks, nheads, head_dim]
        # Avoid atomic add conflict
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         head_dim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    if dz is not None:
        assert dz.shape == z.shape
    else:
        dz = torch.empty_like(z)
    if recompute_output:
        outz = torch.empty_like(x)
    dout_x = torch.empty_like(dout)
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                  if D is not None else (0, 0, 0, 0, 0))
    grid_dz = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dz_kernel[grid_dz](
            dout, out, z, x, D, outz if recompute_output else None,
            dz, dout_x, dD, ddA_cumsum if has_ddAcs else None,
            chunk_size, head_dim,
            batch, seq_len,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            D.stride(0) if D is not None else 0,
            *((outz.stride(0), outz.stride(1), outz.stride(2), outz.stride(3)) if recompute_output else (0, 0, 0, 0)),
            dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3),
            dout_x.stride(0), dout_x.stride(1), dout_x.stride(2), dout_x.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            *((ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3)) if has_ddAcs else (0, 0, 0, 0)),
            D is not None,
            D.dim() == 2 if D is not None else True,
            has_ddAcs,
            BLOCK_SIZE_N=max(triton.next_power_of_2(head_dim), 16),
            RECOMPUTE_OUTPUT=recompute_output,
        )
    if D is not None:
        # BLOCK_SIZE_M
        BLOCK_SIZE_actual = _chunk_scan_bwd_dz.best_config.kwargs['BLOCK_SIZE_M']
        # math.ceil(chunk_size, BLOCK_SIZE_actual)
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return_vals = (dz, dout_x, dD, ddA_cumsum) if has_ddAcs else (dz, dout_x, dD)
    return return_vals if not recompute_output else (*return_vals, outz)

def _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seq_len, nheads, head_dim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, d_state = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seq_len, ngroups, d_state)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    dtype = C.dtype if dtype is None else dtype
    dprev_states = torch.empty(batch, nchunks, nheads, head_dim, d_state, device=C.device, dtype=dtype)
    grid_states = lambda META: (triton.cdiv(head_dim, META['BLOCK_SIZE_M']) * triton.cdiv(d_state, META['BLOCK_SIZE_N']),
                                batch * nchunks, nheads)
    with torch.cuda.device(C.device.index):
        _chunk_scan_bwd_dstates_kernel[grid_states](
            dout, C, dprev_states, dA_cumsum, seq_idx,
            head_dim, d_state, chunk_size,
            batch, seq_len, nchunks, nheads // ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dprev_states.stride(0), dprev_states.stride(1), dprev_states.stride(2), dprev_states.stride(3), dprev_states.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return dprev_states

_CHUNK_SCAN_BWD_DC_MIN_BLOCK_N = min(
    cfg.kwargs['BLOCK_SIZE_N'] for cfg in _chunk_scan_bwd_dc_kernel.configs
)

@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ]),
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_dC_kernel(
    # Pointers to matrices
    dout_ptr, prev_states_ptr, C_ptr, dA_cumsum_ptr, seq_idx_ptr,
    dc_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_prev_states_batch, stride_prev_states_chunk, stride_prev_states_head, stride_prev_states_hdim, stride_prev_states_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dc_batch, stride_dc_seqlen, stride_dc_split, stride_dc_group, stride_dc_dstate,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize, stride_ddA_tile,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = triton.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n

    # Move pointers
    pid_ngroups_splits = pid_g * (nheads // ngroups) + pid_s * nheads_per_program
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_ngroups_splits * stride_dout_head
    dc_ptr += pid_b * stride_dc_batch + pid_c * chunk_size * stride_dc_seqlen + pid_s * stride_dc_split + pid_g * stride_dc_group
    prev_states_ptr += pid_b * stride_prev_states_batch + pid_c * stride_prev_states_chunk + pid_ngroups_splits * stride_prev_states_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_ngroups_splits * stride_dA_cs_head
    if HAS_DDA_CS:
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_ngroups_splits * stride_ddA_cs_head + pid_n * stride_ddA_tile
        C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + pid_g * stride_C_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim
    prev_states_ptrs = prev_states_ptr + offs_k[:, None] * stride_prev_states_hdim + offs_n[None, :] * stride_prev_states_dstate
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    if HAS_DDA_CS:
        C_ptrs = C_ptr + offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c > 0, other=0)
    nheads_per_program_limit = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_per_program_limit):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] * hdim) & (offs_n[None, :] < dstate), other=0.0)
        prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
        # [M, K] @ [K, N] -> [M, N]
        # dc = dout @ prev_states
        dc = tl.dot(dout, prev_states, acc)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        # [M, N] *= [M, 1]
        # dc = prev_states @ dc * scale_m
        dc *= scale_m[:, None]
        if HAS_DDA_CS:
            # [M, N] * [M, N] -> [M,]
            ddA_cs = tl.sum(dc * c, axis=1)
            if DETERMINISTIC_REDUCTION:
                tl.store(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
            else:
                tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
        acc += dc
        # Move pointers to next head
        dout_ptrs += stride_dout_head
        prev_states_ptrs += stride_prev_states_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    dc_ptrs = dc_ptr + offs_m[:, None] * stride_dc_seqlen + offs_n[None, :] * stride_dc_dstate
    tl.store(dc_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


def _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, seq_idx=None, C=None, ngroups=1):  
    batch, nchunks, nheads, head_dim, d_state = prev_states.shape
    _, seq_len, _, _, = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, head_dim, d_state)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seq_len, nheads, head_dim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    deterministic = use_deterministic_mode()
    if C is not None:
        assert C.shape == (batch, seq_len, ngroups, d_state)
        C_strides = (C.stride(0), C.stride(1), C.stride(2), C.stride(3))
        tile_count = math.ceil(d_state / _CHUNK_SCAN_BWD_DC_MIN_BLOCK_N)
        ddA_cumsum_prev, stride_ddA_tile = alloc_tile_workspace(
            (batch, nheads, nchunks, chunk_size),
            tile_count,
            torch.float32,
            dout.device,
            deterministic,
            zero_init=True,
        )
        ddA_cumsum_prev_strides = (
            ddA_cumsum_prev.stride(0),
            ddA_cumsum_prev.stride(2),
            ddA_cumsum_prev.stride(1),
            ddA_cumsum_prev.stride(3),
        )
    else:
        C_strides = (0, 0, 0, 0)
        ddA_cumsum_prev = None
        ddA_cumsum_prev_strides = (0, 0, 0, 0)
        stride_ddA_tile = 0
    nheads_ngroups_ratio = nheads // ngroups
    # SM numbers
    sm_count = torch.cuda.get_device_properties(dout.device).multi_processor_count
    total_nheads = batch * nchunks * nheads
    nheads_per_sm = min(math.ceil(total_nheads / sm_count), nheads_ngroups_ratio)
    nheads_per_program = max(nheads_per_sm, 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dC = torch.empty(batch, seq_len, nsplits, ngroups, d_state, device=dout.device, dtype=torch.float32)
    grid_dc = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(d_state, META['BLOCK_SIZE_N']),
                            batch * nchunks, ngroups * nsplits)
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_dC_kernel[grid_dc](
            dout, prev_states, C, dA_cumsum, seq_idx, dC, ddA_cumsum_prev,
            chunk_size, d_state, head_dim,
            batch, seq_len, nheads, nheads_per_program, ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
            *C_strides,
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3), dC.stride(4),
            *ddA_cumsum_prev_strides, stride_ddA_tile,
            HAS_DDA_CS=ddA_cumsum_prev is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            DETERMINISTIC_REDUCTION=deterministic,
            BLOCK_SIZE_K=max(triton.next_power_of_2(head_dim), 16),
        )
    dC = dC.sum(2)
    if ddA_cumsum_prev is not None:
        ddA_cumsum_prev = finalize_tile_workspace(ddA_cumsum_prev, deterministic)
    return dC if C is None else (dC, ddA_cumsum_prev)