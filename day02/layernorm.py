import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _layer_norm_fwd_fused(
    x, # pointer to the input
    Y, # pointer to the output
    W, # pointer to the weights
    B, # pointer to the bias
    Mean, # pointer to the mean
    Rstd, # pointer to the 1/std
    stride, # stride of the row
    N, # number of columns in X
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    
    X += pid_row * stride
    Y += pid_row * stride

    # compute mean
    mean = 0
    sum_block = tl.float32(0.0)
    for off in tl.arange(0, N, BLOCK_SIZE):
        # [0, 1, 2, ..., BLOCK_SIZE-1]
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        sum_block += tl.sum(a, axis=0)
    mean = sum_block / N

    # compute variance 
    sum_sq = tl.float32(0.0)
    for off in tl.arange(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        sum_sq += tl.sum(x * x, axis=0)
    var = sum_sq / N
    rstd = 1 / tl.math.sqrt(var + eps)
    # write mean and rstd
    tl.store(Mean + pid_row, mean)
    tl.store(Rstd + pid_row, rstd)

    # layernorm
    for off in tl.arange(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        # write ouput
        tl.store(Y + cols, y, mask=mask)



