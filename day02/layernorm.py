import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

"""
Step 1 - unit test
Step 2 - wrapper
Step 3 - forward pass kernel
Step 4 - backward pass kernel
Step 5 - benchmark
"""

###### Step 3 forward pass kernel ######
@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    stride_M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M
    
    # Compute mean
    sum_val = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        
        x_ptrs = tl.load(x_ptr + cols, mask = cols < N, other=0.).to(tl.float32)
        sum_val += tl.sum(x_ptrs, axis=0)
        
    mean = sum_val / N
    
    # Compute variance
    sum_var = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        
        x_ptrs = tl.load(x_ptr + cols, mask = cols < N, other=0.).to(tl.float32)
        diff = tl.where(cols < N, x_ptrs - mean, 0.0)
        sum_var += tl.sum(diff * diff, axis=0)
    var = sum_var / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # write the mean and rstd 
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    
    # Layernorm
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x_ptrs = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32) # [BLOCK_SIZE]
        w_ptrs = tl.load(weight_ptr + cols, mask=mask) # [BLOCK_SIZE]
        b_ptrs = tl.load(bias_ptr + cols, mask=mask) # [BLOCK_SIZE]
        
        x_hat = (x_ptrs - mean) * rstd # [BLOCK_SIZE]
        y = x_hat * w_ptrs + b_ptrs # [BLOCK_SIZE]
        
        # write the output of layernorm
        tl.store(y_ptr + cols, y, mask=mask)

####### Step 4 backward pass kernel ######
@triton.jit
def _layernorm_backforward_dLdx(
    x_ptr, dLdy_ptr, dLdx_ptr, # shape [M, N]
    w_ptr, b_ptr, mean_ptr, rstd_ptr, # shape [N, ] and shape [M]
    dLdw_intermediate_ptr, dLdb_intermediate_ptr, # shape [GROUP_SIZE, N]
    locks_ptr,# shape [2 * GROUP_SIZE]
    stride_M, N,
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride_M
    dLdy_ptr += PID * stride_M
    dLdx_ptr += PID * stride_M
    
    # Load data to SRAM
    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)       # shape (BLOCK_SIZE_N)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0.).to(tl.float32) # shape (BLOCK_SIZE_N)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)                 # shape (BLOCK_SIZE_N)
    b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)                 # shape (BLOCK_SIZE_N) not used
    mean = tl.load(mean_ptr + PID)                                            # shape (1)
    rstd = tl.load(rstd_ptr + PID)                                            # shape (1)
    
    # Compute dLdx
    x_hat = tl.where(cols < N, (x - mean) * rstd, 0.0) # shape (BLOCK_SIZE_N)
    dy_w = tl.where(cols < N, dLdy * w, 0.0)           # shape (BLOCK_SIZE_N)
    c1 = tl.sum(x_hat * dy_w, axis=0) / N                            # shape (1)
    c2 = tl.sum(dLdy * w, axis=0) / N                                # shape (1)
    dLdx = rstd * (dy_w - x_hat * c1 - c2)                           # shape (BLOCK_SIZE_N)
    
    # write dLdx to the DRAM
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)
    
    # Compute the dLdw & dLdb
    dLdw_contribution = (dLdy * x_hat).to(w.dtype) # shape (BLOCK_SIZE_N)
    dLdb_contribution = (dLdy).to(w.dtype)         # shape (BLOCK_SIZE_N)
    
    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    
    # lock [2 * GROUP_SIZE], the first GROUP_SIZE is lock, the second GROUP_SIZE is counting how many accumulations have happended of this lock
    count_ptr = locks_ptr + GROUP_SIZE
    
    # which group we write into
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols    
    
    # old_val = tl.atomic(ptr, expected, val)
    # if *ptr == expected, then *ptr = val, return old
    # else return old
    # if locks_ptr == 1, it means it is locked, we should wait until it is unlocked
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
    
    count = tl.load(count_ptr) # shape (1)
    if count == 0: # if this PID is the first one to access the lock
        tl.atomic_xchg(count_ptr, 1) # 1 means we already have accessed it
    else: # but if this is not the first pid in the accumulation process, we should got the old value from the DRAM and add them to dLdw_contribution.
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask)
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask)
    
    # write the accumulation values to the DRAM
    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)
    
    # release the lock
    tl.atomic_xchg(locks_ptr, 0)

@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr, dLdb_intermediate_ptr, dLdw_ptr, dLdb_ptr, # shape [GROUP_SIZE, N] shape (N)
    GROUP_SIZE, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # accumulate the dLdw & dLdb shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None]
        
        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.)
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)
        
    
    sum_dLdw = tl.sum(dLdw_acc, axis=0) # shape (BLOCK_SIZE_N)
    sum_dLdb = tl.sum(dLdb_acc, axis=0) # shape (BLOCK_SIZE_N)
    
    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)
    
###### Step 2  wrapper #######
class LayerNorm(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx, # ctx is an object we use to store info that'll be used later in the backward pass
            # it doesn't actually get inputted when using .forward(), rather it's handled by the parent class
        x, # input data
        normalized_shape, # not used
        weight, 
        bias,
        eps
    ):
        M, N = x.reshape(-1, x.shape[-1]).shape
        # allocate intermediary tensors and final output
        mean = torch.empty((M, ), dtype=torch.float32, device=DEVICE)
        rstd = torch.empty((M, ), dtype=torch.float32, device=DEVICE)
        y = torch.empty_like(x)
        
        # 1 GPU shared memory is 64 kb, if the features < 64 kb, then we can use kernel
        MAX_FUSED_SIZE = 64 * 1024 // x.element_size()
        
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        _layernorm_forward[(M, )](
            x, y, weight, bias,
            mean, rstd,
            x.stride(0),
            N, # for mask
            eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        
        return y
    
    @staticmethod
    def backward(
        ctx,
        dLdy,
    ):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and 
        we need to compute the gradient of the loss with respect to the input(s).
        """
        # fetcing the original inputs, intermediary tensors, and meta-parameters
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape
        
        # allocate gradients of original inputs 
        dLdw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdb = torch.empty((N, ), dtype=b.dtype, device=b.device)
        dLdx = torch.empty_like(dLdy)
        
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256 
        
        dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=b.device)
        
        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)
        
        _layernorm_backforward_dLdx[(M, )](
            x, dLdy, dLdx,
            w, b, mean, rstd,
            dLdw_intermediate, dLdb_intermediate,
            locks,
            x.stride(0), N,
            GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_N = ctx.BLOCK_SIZE, num_warps = ctx.num_warps
        )
        
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])] # dLdw and dLdb, parallelized by cols
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_intermediate, dLdb_intermediate, dLdw, dLdb,
            min(GROUP_SIZE, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, 
        )
        
        return dLdx, None, dLdw, dLdb, None
    
layernorm = LayerNorm.apply

###### Step 1  unit test #######
def test_layernorm_forward(M, N, dtype, eps=1e-5, device=DEVICE):
    # input data of shape: [M, N]
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    # weight and bias of shape: [N, ]
    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=DEVICE, requires_grad=True)
    # upstream dLdy of shape : [M, N]
    dLdy = .1 * torch.randn_like(x)
    # setting requires_grad to True here instead of x's initial definition means the graph doesn't have to move through 
    #  the -2.3 and 0.5 operations. That's not a big deal here for testing but if we didn't do it in the benchmark then
    #  those results would be confounded by the kernels pytorch implements for entry-wise multiplication and addition
    x.requires_grad_(True)
    
    # forward pass
    y_triton = layernorm(x, (N, ), weight, bias, eps)
    y_ref =  torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps)
    torch.testing.assert_close(y_triton, y_ref, atol=1e-2, rtol=0)
    print("Passed fwd!")
    
    # backward pass (triton)
    y_triton.backward(dLdy, retain_graph=True) # compute x.grad, weight.grad, bias.grad
    #  retain_graph is False by default, if True, save the computational graph
    
    dLdx_triton, dLdw_triton, dLdb_triton = [_.grad.clone() for _ in [x, weight, bias]] # clone means we can detach our grad
    x.grad, weight.grad, bias.grad = None, None, None
    # backward (pytorch)
    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]
    
    # compare
    torch.testing.assert_close(dLdx_triton, dLdx_ref, atol=1e-2, rtol=0)    
    torch.testing.assert_close(dLdw_triton, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_triton, dLdb_ref, atol=1e-2, rtol=0)
    print("Passed bwd!")
    
    
###### Step 5 benchmark ######
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    )
)
def benchmark(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (N, )
    
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    x.requires_grad_(True) 
    quantiles = [0.5, 0.05, 0.95]
    
    def y_fwd():
        if provider == 'triton':
            return layernorm(x, w_shape, weight, bias, eps)
        if provider == 'torch':
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)
        
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500) 
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(dLdy, retain_graph=True), 
            quantiles=quantiles, grad_to_none=[x], rep=500
        )
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    test_layernorm_forward(1151, 8192, torch.float16)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)