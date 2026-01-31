import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat
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
    key=['chunk_size', 'K', 'IS_CAUSAL'],
)
@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    seq_len, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seq_len, stride_a_ngroups, stride_a_K,
    stride_b_batch, stride_b_seq_len, stride_b_ngroups, stride_b_K,
    stride_out_batch, stride_out_nchunks, stride_out_head, stride_out_M, stride_out_N,
    stride_seq_idx_batch, stride_seq_idx_seq_len,
    # Meta-parameter
    IS_CAUSAL: tl.constexpr,
    DOT_TYPE: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_batch = tl.program_id(1)
    pid_nchunks_ngroups = tl.program_id(2)
    pid_chunk = pid_nchunks_ngroups // ngroups
    pid_head = pid_nchunks_ngroups - pid_chunk * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n
    """
    C0B0  0    0
    C1B0 C1B1  0
    C2B0 C2B1 C2B2
    """
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_batch * stride_a_batch + pid_chunk * chunk_size * stride_a_seq_len + pid_head * stride_a_ngroups
    b_ptr += pid_batch * stride_b_batch + pid_chunk * chunk_size * stride_b_seq_len + pid_head * stride_b_ngroups
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_batch * stride_seq_idx_batch + pid_chunk * chunk_size * stride_seq_idx_seq_len

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_a_seq_len + offs_k[None, :] * stride_a_K
    # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    b_ptrs = b_ptr + offs_k[:, None] * stride_b_K + offs_n[None, :] * stride_b_seq_len
    # When the last chunk is smaller than chunk_size
    chunk_size_limit = min(chunk_size, seq_len - pid_chunk * chunk_size)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.).to(DOT_TYPE)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.).to(DOT_TYPE)
        acc += tl.dot(a, b)
        
        # Move pointers
        a_ptrs += BLOCK_SIZE_K * stride_a_K
        b_ptrs += BLOCK_SIZE_K * stride_b_K
        
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seq_len - pid_chunk * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seq_len, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seq_len, mask=offs_n < chunk_size_limit, other=-1)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty) 
    
    out_ptr += pid_batch * stride_out_batch + pid_chunk * stride_out_nchunks + pid_head * stride_out_head
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_M + offs_n[None, :] * stride_out_N
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
    
def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False, output_type=None):
    """
    Input:
    - a: [batch, seq_len, ngroups, d_state]
    - b: [batch, seq_len, ngroups, head_dim]
    - seq_idx: [batch, seq_len]
    - causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Output:
    - out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    has_group = a.dim() == 4
    if not has_group:
        batch, seq_len, k = a.shape
    else:
        batch, seq_len, ngroups, k = a.shape
    assert a.shape == b.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seq_len)
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seq_len / chunk_size)
    out_dtype = a.dtype if output_type is None else output_type
    
    # out: [batch, nchunks, ngroups, chunk_size, chunk_size]
    out = torch.empty((batch, nchunks, chunk_size, chunk_size) if not has_group else (batch, nchunks, ngroups, chunk_size, chunk_size), 
                      device=a.device, dtype=out_dtype)
    dot_type = (tl.bfloat16 if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16 else
                (tl.float16 if a.dtype == torch.float16 or b.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                         batch, nchunks if not has_group else nchunks * ngroups)
    with torch.cuda.device(device=a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a, b, out, seq_idx,
            seq_len, chunk_size, k, ngroups if has_group else 1,
            a.stride(0), a.stride(1), 0 if not has_group else a.stride(2), a.stride(-1),
            b.stride(0), b.stride(1), 0 if not has_group else b.stride(2), b.stride(-1),
            out.stride(0), out.stride(1), 0 if not has_group else out.stride(2), out.stride(-2), out.stride(-1),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            causal,
            dot_type,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out

@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=2),
    ]),
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    # Pointers to matrices
    a_ptr, dout_ptr, db_ptr, res_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    # Meta-parameters
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_seqlen
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)
    dout_ptrs = dout_ptr + offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m
    a_ptrs = a_ptr + offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(a, dout)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen
    
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K), other=0.0).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptrs = db_ptr + offs_m[:, None] * chunk_size * stride_db_seqlen + offs_n[None, :] * stride_db_k
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))

def _bmm_chunk_bwd(a, dout, residual=None, out=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dout: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
        residual: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Return:
        out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)

    If there was seq_idx in the fwd pass, then dout[i, j] for seq_idx[i] != seq_idx[j] should already be
    zeroed out before calling this function.
    """
    has_group = a.dim() == 4
    if not has_group:
        batch, seq_len, k = a.shape
    else:
        batch, seq_len, ngroups, k = a.shape
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        assert residual.shape == (batch, seq_len, k) if not has_group else (batch, seq_len, ngroups, k)
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
                residual = residual.contiguous()
    if out is not None:
        assert out.shape == a.shape
        assert out.stride(-1) == 1 or out.stride(1) == 1 # Make sure out is row-major or column-major
    else:
        out = torch.empty_like(a)
    dot_type = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype == torch.bfloat16 else
                (tl.float16 if a.dtype == torch.float16 or dout.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(k, META['BLOCK_SIZE_N']),
                         batch, nchunks if not has_group else nchunks * ngroups)
    residual_strides = ((residual.stride(0), residual.stride(1), 0 if not has_group else residual.stride(2),
                         residual.stride(-1)) if residual is not None else (0, 0, 0, 0))
    with torch.cuda.device(device=a.device.index):
        _bmm_chunk_bwd_kernel[grid](
            a, dout, out, residual,
            seq_len, chunk_size, k, ngroups if has_group else 1,
            a.stride(0), a.stride(1), 0 if not has_group else a.stride(2), a.stride(-1),
            dout.stride(0), dout.stride(1), 0 if not has_group else dout.stride(2), dout.stride(-2), dout.stride(-1),
            out.stride(0), out.stride(1), 0 if not has_group else out.stride(2), out.stride(-1),
            residual_strides[0], residual_strides[1], residual_strides[2], residual_strides[3],
            dot_type,
            HAS_RESIDUAL=residual is not None,
        )
    return out