

import torch
import triton
import triton.language as tl
from functools import partial

@triton.jit
def grouped_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    # 按照group_size进行分组，重新排列
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)
    
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 矩阵A和矩阵B的偏移
    offset_a = offset_m[:, None] * stride_am + offset_k[None] * stride_ak
    offset_b = offset_k[:, None] * stride_bk + offset_n[None] * stride_bn
    
    c_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, k, BLOCK_SIZE_K):
        a_block = tl.load(a_ptr + offset_a)
        b_block = tl.load(b_ptr + offset_b)
        c_block += tl.dot(a_block, b_block)
        # 移动矩阵A和矩阵B的偏移
        offset_a += BLOCK_SIZE_K * stride_ak
        offset_b += BLOCK_SIZE_K * stride_bk
        
    c_block_ptr = c_ptr + offset_m[:, None] * stride_cm + offset_n[None] * stride_cn
    mask = (offset_m[:, None] < m) & (offset_n[None] < n)
    tl.store(c_block_ptr, c_block, mask=mask)

def matmul(a, b, matmul_k_fn, bs=16, GROUP_SIZE=None):
    # 检查矩阵维度是否兼容
    assert a.shape[1] == b.shape[0], "矩阵维度不兼容，无法进行矩阵乘法"
    # 检查张量是否准备好在 GPU 上运行
    assert a.is_contiguous()
    # 获取矩阵 a 和 b 的形状
    (m, k), (_, n) = a.shape, b.shape
    # 创建一个空的输出张量 c
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    # 定义网格函数，用于计算线程块的数量
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']),  triton.cdiv(n, meta['BLOCK_SIZE_N']))
    # 处理 group_sz 参数，如果为 None，则使用空字典
    GROUP_SIZE = {} if GROUP_SIZE is None else {"GROUP_SIZE":GROUP_SIZE} # 在 naive_matmul 中未使用，但在后续的 grouped_matmul 中会用到
    # 调用 matmul_k_fn 函数，传入必要的参数
    matmul_k_fn[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=bs, BLOCK_SIZE_N=bs, BLOCK_SIZE_K=bs, # 注意：对于较旧的 GPU，allow_tf32 必须设置为 False，否则无法编译
        **GROUP_SIZE
    )
    # 返回计算结果
    return c
grouped_matmul = partial(matmul, matmul_k_fn=grouped_matmul_k)
a = torch.ones((3, 4), dtype=torch.float32, device='cuda')
b = torch.ones((4, 5), dtype=torch.float32, device='cuda')
grouped_matmul(a,b, GROUP_SIZE=4)


import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_warps=8, num_stages=3)],
    key = ['M', 'N', 'K'],
)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_pid = pid // num_pid_in_group
    first_pid_m = group_pid * GROUP_SIZE
    # if BLOCK_SIZE_M can not divided by GROUP_SIZE
    group_size_m = min(GROUP_SIZE, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_block = a_ptr + offset_am[:, None] * stride_am + offset_k[None] * stride_ak
    b_block = b_ptr + offset_k[:, None] * stride_bk + offset_bn[None] * stride_bn
    
    c_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block, mask=((offset_am[:, None] < M) & offset_k[None] < K), other=0.0)
        b = tl.load(b_block, mask=((offset_k[:, None] < K) & (offset_bn[None] < N)), other=0.0)
        
        c_block = tl.dot(a, b, c_block)
        a_block += BLOCK_SIZE_K * stride_ak
        b_block += BLOCK_SIZE_K * stride_bk
        
    if ACTIVATION == "leaky_relu":
        c_block = leaky_relu(c_block)     
    c_block = c_block.to(tl.float16)
    
    
    c_ptrs = c_ptr + offset_am[:, None] * stride_cm + offset_bn[None] * stride_cn
    mask = (offset_am[:, None] < M) & (offset_bn[None] < N)
    tl.store(c_ptrs, c_block, mask=mask)  
    
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

    def triton_matmul(a, b, activation=""):
    # Check constraints.
    # 检查约束
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    # 1 维启动核心，其中每个块都有自己的程序。
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

DATA_TYPE = torch.float16
seq_len = 2048
torch.manual_seed(0)
a = torch.randn((seq_len, seq_len), device='cuda', dtype=DATA_TYPE)
b = torch.randn((seq_len, seq_len), device='cuda', dtype=DATA_TYPE)

triton_matmul(a, b)

# triton_output = triton_matmul(a, b)
# torch_output = torch.matmul(a, b)
# # print(f"triton_output_with_fp16_inputs={triton_output}")
# # print(f"torch_output_with_fp16_inputs={torch_output}")
# print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))