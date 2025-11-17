import triton
import triton.language as tl
import torch

import tabulate
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _seed_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # load data
    x = tl.load(input_ptr + offset, mask=mask)
    
    # generate random numbers
    random = tl.rand(seed, offset)
    x_keep = random > p
    
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offset, output, mask=mask)
    
def seed_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    # How many numbers
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seed_dropout_kernel[grid](
        x,
        output,
        n_elements,
        p,
        seed,
        BLOCK_SIZE=1024
    )
    return output
def test_seed_dropout_kernel():
    x = torch.randn(size=(10, ), device=DEVICE)
    
    output = seed_dropout(x, p=0.5, seed=123)
    output1 = seed_dropout(x, p=0.5, seed=123)
    output2 = seed_dropout(x, p=0.5, seed=512)
    
    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["ouput (seed = 123)"] + output.tolist(),
            ["output (seed = 123)"] + output1.tolist(),
            ["output (seed = 512)"] + output2.tolist(), 
        ])
    )

if __name__ == "__main__":
    test_seed_dropout_kernel()