import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

def softplus(dt):
        return tl.math.log1p(tl.exp(dt))

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

"""
Mathematical Formula:
1. Discretization of dt (Softplus + Clamp):
    $$ Δ_t = clamp(softplus(dt_logits_t + bias), min, max) $$

2. Instantaneous Decay (dA):
    $$ dA_t = Δ_t * A $$  (where A is the continuous decay parameter)

3. Cumulative Decay (Prefix Sum / Integral):

    $$ L_t = \sum_{i=0}^{t} dA_i $$

Variables Mapping:
    - Input dt: dt_logits (Raw network output)
    - Output dt: Δ (Physical time-step)
    - Output dA_cumsum: L (The "Log-Mask" matrix basis)
"""
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_mamba_fwd_kernel(
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    batch, seq_len, nheads, chunk_size,
    dt_min, dt_max,
    stride_dt_batch, stride_dt_seq_len, stride_dt_nheads,
    stride_A_nheads,
    stride_dt_bias_nheads,
    stride_dt_out_batch, stride_dt_out_nchunks, stride_dt_out_nheads, stride_dt_out_chunk_size,
    stride_dA_cumsum_batch, stride_dA_cumsum_nchunks, stride_dA_cumsum_nheads, stride_dA_cumsum_chunk_size,
    
    # Meta-parameter
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_nhead = tl.program_id(2)
    
    dt_ptr += pid_batch * stride_dt_batch + pid_chunk * BLOCK_SIZE_CHUNK * stride_dt_seq_len
    # stride_dt_out_nchunks = BLOCK_SIZE_CHUNK * stride_dt_out_chunk_size = BLOCK_SIZE_CHUNK * stride_dt_seq_len
    dt_out_ptr += pid_batch * stride_dt_out_batch + pid_chunk * stride_dt_out_nchunks
    dA_cumsum_ptr += pid_batch * stride_dA_cumsum_batch + pid_chunk * stride_dA_cumsum_nchunks
    
    # offsets
    offs_head = pid_nhead * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_chunk = tl.arange(0, BLOCK_SIZE_CHUNK)
    # dt_ptrs: [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]
    dt_ptrs = dt_ptr + (offs_head[:, None] * stride_dt_nheads + offs_chunk[None, :] * stride_dt_seq_len)
    # A_ptrs: [BLOCK_SIZE_H]
    A_ptrs = A_ptr + offs_head * stride_A_nheads
    # dt_out_ptrs: [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]
    dt_out_ptrs = dt_out_ptr + (offs_head[:, None] * stride_dt_out_nheads + offs_chunk[None, :] * stride_dt_out_chunk_size)
    # dA_cumsum_ptrs: [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]
    dA_cumsum_ptrs = dA_cumsum_ptr + (offs_head[:, None] * stride_dA_cumsum_nheads + offs_chunk[None, :] * stride_dA_cumsum_chunk_size)
    chunk_size_limit = min(chunk_size, seq_len - pid_chunk * BLOCK_SIZE_CHUNK)
    
    # Load data from HBM to SRAM
    dt = tl.load(dt_ptrs, mask=(offs_head[:, None] < nheads) & (offs_chunk[None, :] < chunk_size_limit), other=0.).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_head * stride_dt_bias_nheads, mask=(offs_head < nheads), other=0.).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1.0), dt)
    
    dt = tl.clamp(dt, dt_min, dt_max)
    # Mask again! Because we did softplus before, if dt is 0, softplus is ln(2)! We need to mask it to 0 again!
    dt = tl.where((offs_head[:, None] < nheads) & (offs_chunk[None, :] < chunk_size_limit), dt, 0.0)
    
    # Write dt to HBM [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] 
    tl.store(dt_out_ptrs, dt, mask=(offs_head[:, None] < nheads) & (offs_chunk[None, :] < chunk_size_limit))
    A = tl.load(A_ptrs, mask=(offs_head < nheads), other=0.).to(tl.float32)
    dA = dt * A[:, None]
    dA_cumsum = tl.cumsum(dA, axis=1)
    # Write dA_cumsum to HBM [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]
    tl.store(dA_cumsum_ptrs, dA_cumsum, mask=(offs_head[:, None] < nheads) & (offs_chunk[None, :] < chunk_size_limit))

def _chunk_mamba_fwd(
    dt,
    A,
    chunk_size,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    batch, seq_len, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seq_len / chunk_size) 
    # dt_out: [batch, nheads, nchunks, chunk_size]
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    # dA_cumsum: [batch, nheads, nchunks, chunk_size]
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    # Parallelize by (batch, nchunks, nheads // BLOCK_SIZE_H)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_mamba_fwd_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_cumsum,
            batch, seq_len, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out

"""
Mathematical Formula:
    For a chunk of length T, the final state h_T is:

    $$ h_T = \sum_{t=0}^{T-1} (X_t)^T \cdot (\bar{B}_t \cdot \text{decay}(t, T)) $$

    Where:
    - Δ_t: Discretized time step at time t
    - L_t: Cumulative decay at time t (from Kernel 1)
    - Decay: w_t = \exp(L_{T-1} - L_t)  (Decay from t to end of chunk)
    - Discretized B: \bar{B}_t = B_t \cdot Δ_t
    
    Simplified Operation (Accumulation Loop):
    acc += X_t^T \cdot (B_t \cdot Δ_t \cdot \exp(L_{last} - L_t))

Variables Mapping:
    - x: V (Value in Attention)
    - b: K (Key in Attention / Input Dynamics)
    - states: The computed chunk state h (HeadDim x DState)
"""
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
    key=['head_dim', 'd_state', 'chunk_size'],
)

@triton.jit
def _chunk_state_fwd_kernel(
    x_ptr, B_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    head_dim, d_state, chunk_size,
    batch, seq_len, nheads_ngroups_ratio,
    stride_x_batch, stride_x_seq_len, stride_x_nheads, stride_x_head_dim,
    stride_B_batch, stride_B_seq_len, stride_B_ngroups, stride_B_d_state,
    stride_states_batch, stride_states_nchunks, stride_states_nheads, stride_states_head_dim, stride_states_d_state,
    stride_dt_batch, stride_dt_nchunks, stride_dt_nheads, stride_dt_chunk_size,
    stride_dA_cs_batch, stride_dA_cs_nchunks, stride_dA_cs_nheads, stride_dA_cs_chunk_size,
    stride_seq_idx_batch, stride_seq_idx_seq_len,
    # Mete Parameter
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch_nchunks = tl.program_id(1)
    pid_chunk = pid_batch_nchunks // batch
    pid_batch = pid_batch_nchunks % batch
    pid_head = tl.program_id(2)
    num_pid_n = tl.cdiv(d_state, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n # head_dim
    pid_n = tl.program_id(0) % num_pid_n # d_state

    # B: [batch, seq_len, ngroups, d_state]
    # X: [batch, seq_len, nheads, head_dim]
    # Similar to GQA(Group Query Attention), 1 group of B is correspond to nheads // ngroups X
    B_ptr += pid_batch * stride_B_batch + pid_chunk * chunk_size * stride_B_seq_len + (pid_head // nheads_ngroups_ratio) * stride_B_ngroups
    x_ptr += pid_batch * stride_x_batch + pid_chunk * chunk_size * stride_x_seq_len + pid_head * stride_x_nheads
    dt_ptr += pid_batch * stride_dt_batch + pid_chunk * stride_dt_nchunks + pid_head * stride_dt_nheads
    dA_cumsum_ptr += pid_batch * stride_dA_cs_batch + pid_chunk * stride_dA_cs_nchunks + pid_head * stride_dA_cs_nheads
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_batch * stride_seq_idx_batch + pid_chunk * chunk_size * stride_seq_idx_seq_len
        
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    x_ptrs = x_ptr + offs_m[:, None] * stride_x_head_dim + offs_k[None, :] * stride_x_seq_len
    # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    B_ptrs = B_ptr + offs_k[:, None] * stride_B_seq_len + offs_n[None, :] * stride_B_d_state
    # [BLOCK_SIZE_K]
    dt_ptrs = dt_ptr + offs_k * stride_dt_chunk_size
    
    chunk_size_limit = min(chunk_size, seq_len - pid_chunk * chunk_size)
    
    # dA_cumsum[..., -1:]
    ####################### What if the last seq idx is smaller than chunk_size - 1 ？ Bug ？############################
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size_limit - 1) * stride_dA_cs_chunk_size).to(tl.float32)
    
    # [BLOCK_SIZE_K]
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_chunk_size
    if HAS_SEQ_IDX:
        # [BLOCK_SIZE_K]
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seq_len
    
    #chunk_size_limit = min(chunk_size, seq_len - pid_chunk * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seq_len)

    # Output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < head_dim) & (offs_k[None, :] < chunk_size_limit), other=0.)
        B = tl.load(B_ptrs, mask=(offs_k[:, None] < chunk_size_limit) & (offs_n[None, :] < d_state), other=0.).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit, other=0.).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit, other=0.).to(tl.float32)
        if not HAS_SEQ_IDX:
            # scale = exp((dA_cs_last - dA_cs_k)) * dt
            scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
        else:
            scale = tl.where((seq_idx_k == seq_idx_last) & (seq_idx_last >= 0),
                             tl.exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k,
                             0.0)
        B *= scale[:, None]
        B = B.to(x_ptr.dtype.element_ty)
        # [BLOCK_SIZE_M, BLOCK_SIZE_K] * [BLOCK_SIZE_K, BLOCK_SIZE_N] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
        acc += tl.dot(x, B)
        
        # Move the pointers
        x_ptrs += BLOCK_SIZE_K * stride_x_seq_len
        B_ptrs += BLOCK_SIZE_K * stride_B_seq_len
        dt_ptrs += BLOCK_SIZE_K * stride_dt_chunk_size
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_chunk_size
        offs_k += BLOCK_SIZE_K
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seq_len
    states = acc.to(states_ptr.dtype.element_ty)
    
    states_ptr += pid_batch * stride_states_batch + pid_chunk * stride_states_nchunks + pid_head * stride_states_nheads
    ######### Why redefine ? ########
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + offs_m[:, None] * stride_states_head_dim + offs_n[None, :] * stride_states_d_state
    tl.store(states_ptrs, states, mask=(offs_m[:, None] < head_dim) & (offs_n[None, :] < d_state))
        
def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    batch, seq_len, nheads, head_dim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, d_state = B.shape
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seq_len, ngroups, d_state)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    # outpu: states: [batch, nchunks, nheads, head_dim, d_state]
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, head_dim, d_state)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, head_dim, d_state), device=x.device, dtype=states_dtype)
    # Parallelize by (head_dim // BLOCK_SIZE_M * d_state // BLOCK_SIZE_N, batch * nchunks, nheads)
    grid = lambda META: (triton.cdiv(head_dim, META['BLOCK_SIZE_M']) * triton.cdiv(d_state, META['BLOCK_SIZE_N']), 
                         batch * nchunks, nheads)
    with torch.cuda.device(device=x.device.index):
        _chunk_state_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, seq_idx,
            head_dim, d_state, chunk_size,
            batch, seq_len, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_state_bwd_db_kernel(
     # Pointers to matrices
    x_ptr, dstates_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    db_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_db_batch, stride_db_seqlen, stride_db_split, stride_db_group, stride_db_dstate,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n
    
    nheads_per_ngroups = nheads // ngroups
    
    # Move pointers to [batch, chunk, head]
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (pid_g * nheads_per_ngroups + pid_g * nheads_per_program) * stride_x_head
    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_s * stride_db_split + pid_g * stride_db_group
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + (pid_g * nheads_per_ngroups + pid_g * nheads_per_program) * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g * nheads_per_ngroups + pid_g * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * nheads_per_ngroups + pid_g * nheads_per_program) * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    if HAS_DDA_CS:
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * nheads_per_ngroups + pid_g * nheads_per_program) * stride_ddA_cs_head
        b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_g * stride_b_head
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_x_seqlen + offs_k[None, :] * stride_x_hdim
    dstates_ptrs = dstates_ptr + offs_k[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    if HAS_DDA_CS:
        b_ptrs = b_ptr + offs_m[:, None] * stride_b_seqlen + offs_n[None, :] * stride_b_dstate
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1))
    nheads_per_program_limit = min(nheads_per_program, nheads_per_ngroups - pid_s * nheads_per_program)
    for h in range(nheads_per_program_limit):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.)
        dstates = dstates.to(x_ptrs.dtype.element_ty)
        # [M, K] * [K, N] -> [M, N] 
        db = tl.dot(x, dstates)
        dA_cumsum_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.).to(tl.float32)
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(tl.minimum(dA_cumsum_last - dA_cs_m, 0.0))
        else:
            scale = tl.where((seq_idx_m == seq_idx_last), 
                             tl.exp(tl.minimum(dA_cumsum_last - dA_cs_m, 0.0)),
                             0.0)
        # db = x @ dstates * scale[:, None] * dt_m[:, None]
        db *= (scale * dt_m)[:, None]
        if HAS_DDA_CS:
            # This is the gradient of wrt (dA_cs_last - dA_cs_m)
            # [M, N] * [M, N] -> [M]
            ddA_cs = tl.sum(db * b, axis=1)
            # if [0, 1, 2, 3], we just need to add to [1, 2, 3], so offset < chunk_size - 1
            """
            [
                0,
                dexp(A1 + A2 + A3),
                dexp(A2 + A3),
                dexp(A3),
            ]
            The gradient of dexp(A1), dexp(A2), dexp(A3) are accumulated to ddA_cs[0], ddA_cs[1], ddA_cs[2] respectively
            [
                0,
                dexp(A1 + A2 + A3),
                dexp(A1 + A2 + A3) + dexp(A2 + A3),
                dexp(A1 + A2 + A3) + dexp(A2 + A3) + dexp(A3),
            ]
            """
            tl.atomic_add(ddA_cumsum_ptrs + stride_dA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1)
        acc += db
        x_ptrs += stride_x_head
        dstates_ptrs += stride_states_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptrs += stride_dA_cs_head
        dA_cumsum_ptr += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    
    db_ptrs = db_ptr + offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_dstate
    tl.store(db_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))
    
    
        
def _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1):
    batch, seq_len, nheads, head_dim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, chunk_size, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    if B is not None:
        assert B.shape == (batch, seq_len, ngroups, dstate)
        B_strides = (B.stride(0), B.stride(1), B.stride(2), B.stride(3))
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=x.dtype)
        ddA_cumsum_strides = (ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3))
    else:
        B_strides = (0, 0, 0, 0)
        ddA_cumsum = None
        ddA_cumsum_strides = (0, 0, 0, 0)
    nheads_ngroups_ration = nheads // ngroups
    # SM numbers
    SM_nums = torch.cuda.get_device_properties(x.device).multi_processor_count
    # Total nheads numbers
    total_nheads = batch * nchunks * nheads
    nheads_per_sm = (total_nheads + SM_nums - 1) // SM_nums
    nheads_per_program = max(1, min(nheads_per_sm, nheads_ngroups_ration))
    # Kernel numbers of per ngroups
    nsplits = triton.cdiv(nheads_ngroups_ration, nheads_per_program)
    dB = torch.empty(batch, seq_len, nsplits, ngroups, dstate, device=x.device, dtype=torch.float32)
    grid_db = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                            batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(device=x.device.index):
        _chunk_state_bwd_db_kernel[grid_db](
            x, dstates, B, dt, dA_cumsum, seq_idx, dB, ddA_cumsum,
            chunk_size, dstate, head_dim,
            batch, seq_len, nheads, nheads_per_program, ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            *B_strides,
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3), dB.stride(4),
            *ddA_cumsum_strides,
            HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(head_dim), 16),
        )
    # [batch, seq_len, nsplits, ngroups, dstate] -> [batch, seq_len, ngroups, dstate]
    dB = dB.sum(2)
    if ddA_cumsum is not None:
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return dB if B is None else (dB, ddA_cumsum)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 2}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 4}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 8}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 16}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 32}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 64}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_bwd_kernel(
       # Pointers to matrices
    ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr, dt_bias_ptr,
    ddt_ptr, dA_ptr, ddt_bias_ptr,
    # Matrix dimensions
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_ddA_batch, stride_ddA_chunk, stride_ddA_head, stride_ddA_csize,
    stride_ddt_out_batch, stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_ddt_batch, stride_ddt_seqlen, stride_ddt_head,
    stride_dA_head,
    stride_ddt_bias_head,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Move pointers
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * chunk_size * stride_ddt_seqlen
    
    # Offsets
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize
    ddA_ptrs = ddA_ptr + offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize
    dt_ptrs = dt_ptr + offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    ddt_ptrs = ddt_ptr + offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    ddA = tl.load(ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.).to(tl.float32)
    ddt_out = tl.load(ddt_out_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit)).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    # Recompute clamp(softplus(dt + dt_bias)) if needed
    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        # y=softplus(x) => dy/dx = sigmoid(x)
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        tl.atomic_add(ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads)
    

def _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")), ddt=None):
    batch, seq_len, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    assert ddA.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt_out.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
    else:
        ddt_bias = None
    if ddt is None:
        ddt = torch.empty_like(dt)
    else:
        assert ddt.shape == dt.shape
    dA = torch.empty_like(A, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid_chunk_cs](
            ddA, ddt_out, dt, A, dt_bias, ddt, dA, ddt_bias,
            batch, seq_len, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            ddA.stride(0), ddA.stride(2), ddA.stride(1), ddA.stride(3),
            ddt_out.stride(0), ddt_out.stride(2), ddt_out.stride(1), ddt_out.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            ddt.stride(0), ddt.stride(1), ddt.stride(2),
            dA.stride(0),
            ddt_bias.stride(0) if ddt_bias is not None else 0,
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return ddt, dA, ddt_bias