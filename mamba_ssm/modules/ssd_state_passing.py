import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

"""
Input:
- states: [batch, nchunks, nheads, dim]
- dA_chunk_cumsum: [batch, nheads, nchunks] The last dA_cumsum of every chunk
- initial_states: [batch, nheads, dim]
- seq_idx: [batch, seq_len]

Output:
- out: [batch, nchunks, nheads, dim]
- final_states: [batch, nheads, dim]

Mathematical Formula:
$$ h_{t} =  h_{t-1} \cdot exp(L_{t}) + h_{t-1}^\prime $$
"""
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': 64}),
             triton.Config({'BLOCK_SIZE': 128}),
             triton.Config({'BLOCK_SIZE': 256}),
             triton.Config({'BLOCK_SIZE': 512}),
             triton.Config({'BLOCK_SIZE': 1024}),
             triton.Config({'BLOCK_SIZE': 2048}),
            ],
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    states_ptr, out_ptr, final_states_ptr, dA_chunk_cumsum_ptr, initial_states_ptr, seq_idx_ptr,
    dim, nchunks, seq_len, chunk_size,
    stride_states_batch, stride_states_nchunks, stride_states_nheads, stride_states_dim,
    stride_out_batch, stride_out_nchunks, stride_out_nheads, stride_out_dim,
    stride_final_states_batch, stride_final_states_nheads, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_nchunks, stride_dA_cs_nheads,
    stride_initial_states_batch, stride_initial_states_nheads, stride_initial_states_dim,
    stride_seq_idx_batch, stride_seq_idx_seq_len,
    # Meta-parameter
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid_nhead = tl.program_id(2)
    pid_dim = tl.program_id(0)
    
    states_ptr += pid_batch * stride_states_batch + pid_nhead * stride_states_nheads
    out_ptr += pid_batch * stride_out_batch + pid_nhead * stride_out_nheads
    final_states_ptr += pid_batch * stride_final_states_batch + pid_nhead * stride_final_states_nheads
    dA_chunk_cumsum_ptr += pid_batch * stride_dA_cs_batch + pid_nhead * stride_dA_cs_nheads
    if HAS_INITSTATES:
        initial_states_ptr += pid_batch * stride_initial_states_batch + pid_nhead * stride_initial_states_nheads
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_batch * stride_seq_idx_batch
    
    offs_dim = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_dim * stride_states_dim
    out_ptrs = out_ptr + offs_dim * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_dim * stride_final_states_dim

    # The last state $$ h_{t-1} $$
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        initial_states_ptrs = initial_states_ptr + pid_dim * stride_initial_states_dim
        states = tl.load(initial_states_ptrs, mask=offs_dim < dim, other=0.).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_dim < dim)
    out_ptrs += stride_out_nchunks
    seq_idx = 0
    for c in range(0, nchunks):
        new_states = tl.load(states_ptrs, mask=offs_dim < dim, other=0.).to(tl.float32) # [BLOCK_SIZE]
        dA_cs = tl.load(dA_chunk_cumsum_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        
        ###### What if this chunk includes two different sequence tokens? ######
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seq_len) - 1) * stride_seq_idx_seq_len)
            scale = tl.where((seq_idx_new == seq_idx), scale, 0.0)
            seq_idx = seq_idx_new
        # [BLOCK_SIZE] = [1] * [BLOCK_SIZE] + [BLOCK_SIZE]
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_dim < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_dim < dim)
        states_ptrs += stride_states_nchunks
        dA_chunk_cumsum_ptr += stride_dA_cs_nchunks
        out_ptrs += stride_out_nchunks

        

def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None, out_dtype=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seq_len = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seq_len)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, seq_idx,
            dim, nchunks, seq_len if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2)) if initial_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out, final_states
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_bwd_kernel(
    dout_ptr, out_ptr, dA_cumsum_ptr, dfinal_states_ptr, seq_idx_ptr,
    dstates_ptr, ddA_chunk_cumsum_ptr, dinitial_states_ptr, states_converted_ptr,
    dim, nchunks, seq_len, chunk_size,
    stride_dout_batch, stride_dout_nchunks, stride_dout_nheads, stride_dout_dim,
    stride_out_batch, stride_out_nchunks, stride_out_nheads, stride_out_dim,
    stride_dA_cs_batch, stride_dA_cs_nchunks, stride_dA_cs_nheads,
    stride_dfinal_states_batch, stride_dfinal_states_nheads, stride_dfinal_states_dim,
    stride_seq_idx_batch, stride_seq_idx_seq_len,
    stride_dstates_batch, stride_dstates_nchunks, stride_dstates_nheads, stride_dstates_dim,
    stride_ddA_cs_batch, stride_ddA_cs_nchunks, stride_ddA_cs_nheads,
    stride_dinitial_states_batch, stride_dinitial_states_nheads, stride_dinitial_states_dim,
    # Meta-parameter
    CONVERT_STATES: tl.constexpr,
    HAS_DFINAL_STATES: tl.constexpr,
    HAS_DINITIAL_STATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_m = tl.program_id(0)
    # Move pointers to the last chunk
    dstates_ptr += pid_b * stride_dstates_batch + pid_h * stride_dstates_nheads + (nchunks - 1) * stride_dstates_nchunks
    # ddA_chunk_cumsum is [batch, nchunks, nheads, n_blocks]
    ddA_chunk_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_h * stride_ddA_cs_nheads + (nchunks - 1) * stride_ddA_cs_nchunks + pid_m * 1
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_nheads + (nchunks - 1) * stride_dA_cs_nchunks
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_nheads + (nchunks - 1) * stride_out_nchunks
    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_nheads + (nchunks - 1) * stride_dout_nchunks
    if CONVERT_STATES:
        states_converted_ptr += pid_b * stride_out_batch + pid_h * stride_out_nheads + (nchunks - 1) * stride_out_nchunks
    if HAS_DFINAL_STATES:
        dfinal_states_ptr += pid_b * stride_dfinal_states_batch + pid_h * stride_dfinal_states_nheads
    if HAS_DINITIAL_STATES:
        dinitial_states_ptr += pid_b * stride_dinitial_states_batch + pid_h * stride_dinitial_states_nheads
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
    """
    dout: [batch, nchunks, nheads, dim]
    dA_chunk_cs: [batch, nchunks, nheads]
    dstates: [batch, nchunks, nheads, dim]
    dstates[i-1] = dA_chunk_cs[i] * dout[i]
    """
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim

    if HAS_DFINAL_STATES:
        dstates = tl.load(dfinal_states_ptr + offs_m * stride_dfinal_states_dim, mask=offs_m < dim, other=0.).to(tl.float32)
    else:
        dstates = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
    if HAS_SEQ_IDX:
        # the last token
        seq_idx = tl.load(seq_idx_ptr + (seq_len - 1) * stride_seq_idx_seq_len)
    # the second last chunk
    dstates_ptrs -= 1 * stride_dstates_nchunks
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cumsum_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            # the last token of the last chunk
            # if now chunk is 4, seq_idx_new = the last token of chunk 3
            seq_idx_new = tl.load(seq_idx_ptr + ((nchunks - c - 1) * chunk_size - 1) * stride_seq_idx_seq_len)
            scale = tl.where((seq_idx == seq_idx_new), scale, 0.0)
            seq_idx = seq_idx_new
        # [BLOCK_SIZE]
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        # $ddA_{cs}[i] = \sum(\sum(dout[i] * out[i-1])) * dA_{cs}[i]$
        # we need to reduction at the head_dim and d_state dim, but we already merge this two dims to 1 dim, so we just need to reduction in the "dim" dim,
        # [BLOCK_SIZE] * [BLOCK_SIZE]
        ddA = tl.sum(dstates * out, axis=0) * scale
        tl.store(ddA_chunk_cumsum_ptr, ddA, mask=offs_m < dim)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.).to(tl.float32)
        # [BLOCK_SIZE]
        # $dout[i-1]=dout[i] * dA_{cs}[i]$
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        dout_ptrs -= stride_dout_nchunks
        out_ptrs -= stride_out_nchunks
        dstates_ptrs -= stride_dstates_nchunks
        ddA_chunk_cumsum_ptr -= stride_ddA_cs_nchunks
        dA_cumsum_ptr -= stride_dA_cs_nchunks
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_dim
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=offs_m < dim)
    if not HAS_DINITIAL_STATES:
        tl.store(ddA_chunk_cumsum_ptr, 0.0)
    else:
        dA_cs = tl.load(dA_cumsum_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            scale = tl.where((seq_idx == 0), scale, 0.0)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.).to(tl.float32)
        ddA = tl.sum(out * dstates, axis=0) * scale
        tl.store(ddA_chunk_cumsum_ptr, ddA, mask=offs_m < dim)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dinitial_states_ptr + offs_m * stride_dinitial_states_dim, dstates, mask=offs_m < dim)


def _state_passing_bwd(
    states, dA_chunk_cumsum, dout, dfinal_states=None, seq_idx=None, has_initial_states=None,
    dstates_dtype=None, states_dtype=None, chunk_size=None,
):
    batch, nchunks, nheads, dim = states.shape
    assert  dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == (batch, nchunks, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seq_len = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seq_len)
    dstates = torch.empty_like(dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(states, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    if has_initial_states:
        dinitial_states = torch.empty_like(dstates[:, 0])
    else:
        dinitial_states = None
    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, dim)
    BLOCK_SIZE_min = 64
    n_blocks = (dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min # match.ceil(dim, BLOCK_SIZE_min)
    ddA_chunk_cumsum = torch.empty(batch, nheads, nchunks, n_blocks, dtype=dA_chunk_cumsum.dtype, device=dA_chunk_cumsum.device)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](
            dout, states, dA_chunk_cumsum, dfinal_states, seq_idx,
            dstates, ddA_chunk_cumsum, dinitial_states, states_converted,
            dim, nchunks, seq_len if seq_idx is not None else 0, chunk_size if chunk_size is not None else 0,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((dfinal_states.stride(0), dfinal_states.stride(1), dfinal_states.stride(2)) if dfinal_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3),
            ddA_chunk_cumsum.stride(0), ddA_chunk_cumsum.stride(2), ddA_chunk_cumsum.stride(1),
            *((dinitial_states.stride(0), dinitial_states.stride(1), dinitial_states.stride(2)) if dinitial_states is not None else (0, 0, 0)),
            CONVERT_STATES=states_converted is not None,
            HAS_DFINAL_STATES=dfinal_states is not None,
            HAS_DINITIAL_STATES=dinitial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
        
    BLOCK_SIZE_actual = _state_passing_bwd_kernel.best_config.kwargs['BLOCK_SIZE']
    n_valid_blocks = (dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = ddA_chunk_cumsum[:, :, :, :n_valid_blocks]
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (dstates, ddA_chunk_cumsum, dinitial_states) if states_dtype is None else (states_converted, ddA_chunk_cumsum, dinitial_states, states_converted)