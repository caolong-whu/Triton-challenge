from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import triton
import triton.language as tl

from ssd_chunk_state import _chunk_mamba_fwd, _chunk_state_fwd, _chunk_state_bwd_db
from ssd_state_passing import _state_passing_fwd, _state_passing_bwd
from ssd_bmm import _bmm_chunk_fwd
from ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates, _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    batch, seq_len, nheads, head_dim = x.shape
    _, _, ngroups, d_state = B.shape
    
    ## assert ##
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seq_len, ngroups, d_state)
    assert x.shape == (batch, seq_len, nheads, head_dim)
    assert dt.shape == (batch, seq_len, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, head_dim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, head_dim, d_state)
    # Step 1: Calculate dA_cumsum, dt
    # # dA_cumsum: [batch, nheads, nchunks, chunk_size]
    # # dt: [batch, nheads, nchunks, chunk_size]
    dA_cumsum, dt = _chunk_mamba_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    # Step 2: Calculate (B * decay).T @ X
    # states: (batch, nchunks, nheads, head_dim, d_state)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    # Step 3: Pass states dA_cumsum * states
    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                              initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                              seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    # states: [batch, nchunks, nheads, head_dim, d_state]
    # final_states: [batch, nheads, head_dim, d_state]
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=d_state) for t in [states, final_states]]
    # Step 4: Calculate C @ B
    # CB: [batch, nchunks, ngroups, chunk_size, chunk_size]
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_type=torch.float32)
    # Step 5: Calculate C * states * dA_cumsm + CB * dA_cumsum * dt * x + D * x
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum, states, final_states


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
    b_ptr, dstates_ptr,
    dx_ptr, ddt_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(2)
    
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n
    
    # Move the ptr to corresponding [batch, chunk, head]
    # x, dout, dx: (B, L, H, P)
    x_ptr = x_ptr + pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr = dout_ptr + pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dx_ptr = dx_ptr + pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head

    # dt, ddt, dA_cumsum: (B, C, H, Q)
    dt_ptr = dt_ptr + pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    ddt_ptr = ddt_ptr + pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head

    # B: [B, L, ngroups, N], CB: [B, C, ngroups, chunk_size, chunk_size] need to consider GQA(Group Query Attention)
    # group_idx = pid_h // ration
    group_idx = pid_h // nheads_ngroups_ratio
    b_ptr = b_ptr + pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + group_idx * stride_b_head
    cb_ptr = cb_ptr + pid_b * stride_cb_batch + pid_c * stride_cb_chunk + group_idx * stride_cb_head
    
    # dstates: [B, C, H, P, N]
    dstates_ptr = dstates_ptr + pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    
    if HAS_SEQ_IDX:
        seq_idx_ptr = seq_idx_ptr + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        
    # ---Offsets---
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # decay scale = exp(dA_end - dA_M)
    # ----------- 1. dx from state formula: state_end_k = (dAcs[end] - dAcs[k])*B_k * x_k -----------
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize,
                      mask=offs_m < chunk_size_limit, other=0.).to(tl.float32)
    # Why not dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize, other=0.).to(tl.float32) ?????
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize, other=0.).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_m, 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen, other=-1)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum(dA_cs_last - dA_cs_m, 0.0)), 0.0)
        
    # B:[cs, N] -> [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # dstates: [P, N] -> [BLOCK_SIZE_N, BLOCK_SIZE_K]
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    
    b_ptrs = b_ptr + offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate
    dstates_ptrs = dstates_ptr + offs_dstate[:, None] * stride_dstates_dstate + offs_n[None, :] * stride_dstates_hdim
    
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        # [M, K] * [K, N] * [M,] -> [M, N]
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < (dstate - k)), other=0.)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < (dstate - k)) & (offs_n[None, :] < hdim), other=0.)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            # [M, K] * [K, N] * [M,] -> [M, N]
            acc += tl.dot(b, dstates)

            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # -----------------------------2. dx from intra-chunk scan: y_k = CB_k * dAcs * mask * x_k --------------------------------
    # start of the loop
    K_MIN = pid_m * BLOCK_SIZE_M
    # end of the loop
    K_MAX = chunk_size_limit
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # cb: [cs_m, cs_k] -> [M, K]
    # stride_cb_csize_m = 1, stride_cb_csize_k = cb.stride(-2)  !!!!!!! so actualy we load the (cb).T to [M, K]!!!!!
    cb_ptrs = cb_ptr + offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    # dout: [cs, P] -> [K, N]
    dout_ptrs = dout_ptr + offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
    # dA_cs: [cs, ] -> [K, ]
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    # move ptrs to K_MIN
    # dx = (CB).T * mask.T * dout
    # the gradient of the x_k is from dyk, dyk+1, dyk+2, ....
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        
        load_mask_k = (offs_k + k < K_MAX)
        
        # cb: [M, K]
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (load_mask_k[None, :]), other=0.)
        # dout: [K, N]
        dout = tl.load(dout_ptrs, mask=(load_mask_k[:, None] & (offs_n[None, :] < hdim)), other=0.)
        # dA_cs_k: [K, ]
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=load_mask_k, other=0.)
        
        # decay = exp(dA_k - dA_m)
        scale_term = tl.exp(tl.minimum(dA_cs_k[None, :] - dA_cs_m[:, None], 0.0))
        # [M, K] * [M, K] -> [M, K]
        cb *= scale_term
        
        # make sure k>=m
        # [M, K]
        mask_causal = (k + offs_k[None, :] >= offs_m[:, None])
        
        mask_combined = mask_causal & (k + offs_k[None, :] < K_MAX)
        
        cb = tl.where(mask_combined, cb, 0.0)
        
        cb = cb.to(dout.dtype.element_ty)
        # [M, K] * [K, N]->[M, N]
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    
    
    # --------------------------3. dx, ddt, dD -------------------------------------
    # u = dt * x ----> dx = du * dt = acc * dt
    # Actually B is not discretized, acc = dL / d(x * dt) !
    
    mask_mn = (offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # [M, ]
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.)
    
    # [M, N] *[M, 1] -> [M, N]
    dx = acc * dt_m[:, None]
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen) + (offs_n[None, :] * stride_dx_hdim)
    # ------ dx from : out = SSM(x) + Dx -----------
    # dx = dx + dout * D
    if HAS_D:
        dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen) + (offs_n[None, :] * stride_dout_hdim)
        # dout_res: [M, N]
        dout_res = tl.load(dout_ptrs, mask=mask_mn, other=0.).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head)
        # [M, N] * [N, ] or [M, N] * [1]
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=mask_mn)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen) + (offs_n[None, :] * stride_x_hdim)
    # x: [M, N]
    x = tl.load(x_ptrs, mask=mask_mn, other=0.).to(tl.float32)
    if HAS_D:
        # dD = sum(dout * x)
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            # [N]
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            # [M, N] * [M, N] -> [N,]
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            # [M, N] * [M, N] -> [1]
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    # u = dt * x --> ddt = sum(acc * x)
    # [M, N] * [M, N] -> [M, ]
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)
            
def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seq_len, nheads, head_dim = x.shape
    _, _, ngroups, d_state = B.shape
    _, _, nchunks, chunk_size = dt.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seq_len, ngroups, d_state)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, head_dim, d_state)
    if D is not None:
        assert D.shape == (nheads, head_dim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         head_dim if D.dim() == 2 else 1, device=D.device, dtype=D.dtype)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                   if D is not None else (0, 0, 0, 0, 0))
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=dt.dtype)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(head_dim, META['BLOCK_SIZE_N']),
                            batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, dx, ddt, dD,
            chunk_size, head_dim, d_state,
            batch, seq_len, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1),CB.stride(-2), #####  We need the transpose of cb so CB.stride(-1),CB.stride(-2) !!!
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(d_state), 16),
            IS_TRITON_22=TRITON_22
        )
        if D is not None:
            BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs['BLOCK_SIZE_M']
            n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
            dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
            if D.dim() == 1:
                dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD

    


def _mamba_chunk_scan_combined_bwd(
    dout, x, dt, A, B, C, out, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, dfinal_states=None, seq_idx=None, dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    dx=None, ddt=None, dB=None, dC=None, dz=None, recompute_output=False
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seq_len, nheads, head_dim = x.shape
    nchunks = math.ceil(seq_len / chunk_size)
    _, _, ngroups, d_state = B.shape
    assert dout.shape == (batch, seq_len, nheads, head_dim)
    assert dt.shape == (batch, seq_len, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seq_len, ngroups, d_state)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, head_dim, d_state)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
    if dC is not None:
        assert dC.shape == C.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    ######## Forward #########
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_mamba_fwd(dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_type=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, _ = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                              initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                              seq_idx=seq_idx, chunk_size=chunk_size)
    states = rearrange(states, "... (p n) -> ... p n", n=d_state)

    ######## Backward ########
    # 1. out = y * SiLU(z) + D * x -> dz, dy, dD
    if z is not None:
        # (dz, dout_x, dD, ddA_cumsum) or (dz, dout_x, dD)
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz, recompute_output=recompute_output)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
        
    # 2. $$y_{t,p} = \underbrace{\text{Decay}_t}_{\text{标量}} \cdot \sum_{n=1}^{N} (\underbrace{h_{p,n}}_{\text{状态}} \cdot \underbrace{C_{t,n}}_{\text{参数}})$$
    
    # $\partial h_{prev}$ 
    # [batch, nchunks, nheads, head_dim, d_state]
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)

    # dstates: [batch, nchunks, nheads, head_dim, d_state]
    # ddA_chunk_cumsum: [batch, nchunks, nheads]
    # dinitial_states: [batch, nheads, head_dim, d_state]
    # states: [batch, nchunks, nheads, head_dim, d_state]
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1], # dA_cumsum: [batch, nheads, nchunks, chunk_size]
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)") if dfinal_states is not None else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    
    states = rearrange(states, "... (p n) -> ... p n", n=d_state)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=d_state)
    dinitial_states = rearrange(dinitial_states, "... (p n) -> ... p n", n=d_state) if dinitial_states is not None else None
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx)
    
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)
    dC, ddA_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups)
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    
    
class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        return_varlen_states=False,
    ):
        ctx.dt_type = dt.dtype
        if not return_varlen_states:
            cu_seqlens = None
        else:
            assert cu_seqlens is not None, "cu_seqlens must be provided if return_varlen_states is True"
        # out is the output after SiLU gated mechansim, out_x is the output of SSM
        out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.return_final_states = return_final_states
        ctx.return_varlen_states = return_varlen_states
        if not return_varlen_states:
            return out if not return_final_states else (out, final_states)
        else:
            varlen_states = rest[0]
            return (out, varlen_states) if not return_final_states else (out, final_states, varlen_states)
    
    @staticmethod
    def backward(ctx, dout, *args):
        
        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
        assert not ctx.return_varlen_states, "return_varlen_states is not supported for backward"
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limi=ctx.dt_limit)
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states, None, None, None, None, None, None
        
        
def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    return_varlen_states=False
):
    """
    Docstring for mamba_chunk_scan_combined
    
    :param x: [B, L, nheads, headdim]
    :param dt: [B, L, nheads]
    :param A: [nheads]
    :param B: [B, L, ngroups, d_state]
    :param C: [B, L, ngroups, d_state]
    :param chunk_size: int
    :param D: [nheads]
    :param z: [B, L, nheads, headdim]
    :param dt_bias: [nheads]
    :param initial_states: [B, nheads, headdim, d_state]
    :param seq_idx: [B, L]
    :param cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
    :param dt_softplus: Whether to apply softplus to dt
    
    :out: [B, L, nheads, headdim]
    """
    return MambaChunkScanCombinedFn.apply(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        seq_idx,
        cu_seqlens,
        dt_softplus,
        dt_limit,
        return_final_states,
        return_varlen_states,
    )