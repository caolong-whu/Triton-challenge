from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import triton
import triton.language as tl

from ssd_chunk_state import _chunk_mamba_fwd, _chunk_state_fwd
from ssd_state_passing import _state_passing_fwd, _state_passing_bwd
from ssd_bmm import _bmm_chunk_fwd
from ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates

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