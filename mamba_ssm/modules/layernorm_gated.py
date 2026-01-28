import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange

@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    # Step 1: compute mean and var
    cols = tl.arange(0, BLOCK_N)
    # x: [BLOCK_N]
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        x_bar = tl.where(cols < N, x - mean, 0.0) # X_i - mean
        v_bar = tl.sum(x_bar * x_bar, axis=0) / N # sum (X_i - mean)^2 / N
    else:
        x_bar = tl.where(cols < N, x, 0.0)
        v_bar = tl.sum(x_bar * x_bar, axis=0) # sum x^2 / N
    rstd = 1 / tl.sqrt(v_bar + eps) # rstd = 1 / (sum x^2 / N)
    tl.store(Rstd + row, rstd)
    
    # Step 2: normalization
    w = tl.load(W + cols, mask=cols < N, other=0.).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=cols < N, other=0.).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=cols < N)
    
def _layer_norm_fwd(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.shape == (M, N)
        assert z.stride(-1) == 1
    if bias is not None:
        assert bias.shape == (N,)
        assert bias.stride(-1) == 1
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
    assert out.stride(-1) == 1
    
    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    
    # Make sure the data of feature dim is smaller than 64kb
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps = max(min(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
            x, out, weight, bias, z, mean, rstd,
            x.stride(0), out.stride(0), z.stride(0) if z is not None else 0,
            M, group_size, eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out, mean, rstd
    
    

class LayernormFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(x, weight, bias, eps, z=z, group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm)
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)
    
    @staticmethod
    def backward(ctx, dy):
        pass


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    return LayernormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, True)

class RMSNormGated(torch.nn.Module):
    
    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
    
    def forward(self, x, z=None):
        return rmsnorm_fn(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                          norm_before_gate=self.norm_before_gate)