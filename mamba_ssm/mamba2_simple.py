import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

BATCH_SIZE = 2
SEQ_LEN = 128
N_HEADS = 4
D_HEAD = 256
D_STATE = 64
CHUNK_SIZE = 64
N_CHUNKS = SEQ_LEN // CHUNK_SIZE

X = torch.randn((BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_HEAD), dtype=torch.float32)
A = torch.randn((BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS), dtype=torch.float32)
B = torch.randn((BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_STATE), dtype=torch.float32)
C = torch.randn((BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_STATE), dtype=torch.float32)


def segsum(x):
    # x: [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE]
    Q = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones((Q, Q), device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill_(~mask, -torch.inf)
    """
    x_segsum: [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, CHUNK_SIZE]
    0,       -inf,     -inf
    A1,       0,       -inf
    A1 + A2,  A2,        0
    """
    return x_segsum

def ssd(X, A, B, C, chunk_size=CHUNK_SIZE, initial_states=None):
    """
    Args:
        X: [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_HEAD]
        A: [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS]
        B: [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_STATE]
        C: [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_STATE]
    Returns:
        Y: [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_HEAD]
    """
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    
    # 1. Compute the 1-ssd output. Orange blocks
    L = torch.exp(segsum(A))
    # Y_diag = L * (C @ B_T) @ X 
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, X)
    
    # Detailed implementation
    # [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_STATE] -> [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_STATE]
    B_permute = B.permute(0, 3, 1, 2, 4)
    C_permute = C.permute(0, 3, 1, 2, 4)
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_STATE] @ [BATCH_SIZE, N_HEADS, N_CHUNKS, D_STATE, CHUNK_SIZE] -> [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, CHUNK_SIZE]
    C_BT = C_permute @ B_permute.transpose(-1, -2)
    
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, CHUNK_SIZE]
    C_BT = L * C_BT 
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, CHUNK_SIZE] @ [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_HEAD] -> [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_HEAD]
    Y_diag_detailed = C_BT @ X.permute(0, 3, 1, 2, 4)
    
    # compare two implementations
    Y_diag_detailed = Y_diag_detailed.permute(0, 2, 3, 1, 4)
    assert torch.allclose(Y_diag, Y_diag_detailed, atol=1e-5), "Y_diag and Y_diag_detailed are not close!"
    
    # 2. Green blocks
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE]
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
    # [BATCH_SIZE, N_CHUNKS, N_HEADS, D_HEAD, D_STATE]
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, X)
    
    # Detailed implementation
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_STATE] * [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, 1] -> [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_STATE]
    B_decayed = B.permute(0, 3, 1, 2, 4) * decay_states[..., None]
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, D_STATE, CHUNK_SIZE] * [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE, D_HEAD] -> [BATCH_SIZE, N_HEADS, N_CHUNKS, D_STATE, D_HEAD]
    states_detailed = B_decayed.transpose(-1, -2) @ X.permute(0, 3, 1, 2, 4)
    states_detailed = states_detailed.permute(0, 2, 1, 4, 3)
    # compare two implementations
    assert torch.allclose(states, states_detailed, atol=1e-5), "states and states_detailed are not close!"
    
    # 3. Yellow blocks
    if initial_states is None:
        # [BATCH_SIZE, 1, N_HEADS, D_HEAD, D_STATE]
        initial_states = torch.zeros_like(states[:, :1])
    # [BATCH_SIZE, N_CHUNKS + 1, N_HEADS, D_HEAD, D_STATE]
    states = torch.cat([initial_states, states], dim=1)
    # [BATCH_SIZE, N_HEADS, N_CHUNKS + 1, N_CHUNKS + 1]
    decay_chunks = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))
    # [BATCH_SIZE, N_CHUNKS + 1, N_HEADS, D_HEAD, D_STATE]
    states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunks, states)
    # states: [BATCH_SIZE, N_CHUNKS, N_HEADS, D_HEAD, D_STATE]
    # final_states: [BATCH_SIZE, 1, N_HEADS, D_HEAD, D_STATE]
    states, final_states = states[:, :-1], states[:, -1:]
    
    # 4. Blue blocks
    # [BATCH_SIZE, N_HEADS, N_CHUNKS, CHUNK_SIZE]
    A_cumsum_out = torch.exp(A_cumsum)
    # [BATCH_SIZE, N_CHUNKS, CHUNK_SIZE, N_HEADS, D_HEAD]
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, A_cumsum_out)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    print("state_decay_out shape: ", A_cumsum_out.shape)
    print("Y_off shape: ", Y_off.shape)
    print("Y Shape: ", Y.shape)

if __name__ == "__main__":
    ssd(X, A, B, C)