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