import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from layernorm_gated import RMSNormGated
except ImportError:
    RMSNormGated = None


from ssd_combined import mamba_chunk_scan_combined

class Mamba2Simple(nn.Module):
    
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        # 2 * (2 * d_model) + 2 * (d_state) + nheads
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        
        # 1D Convolution xBC: [B, L, d_inner + 2 * ngroups * d_state]
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.d_conv,
            bias=conv_bias,
            groups=conv_dim,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, a=-self.conv_init, b=self.conv_init)
        
        if self.learnable_init_states:
            self.initial_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.initial_states._no_weight_decay = True
        
        self.act = nn.SiLU()
        
        ## Initialize log dt bias ##
        # log uniform distribution
        # [0.001, 0.1] -> [log(0.001), log(0.1)] 
        # the probability of [0.001, 0.01] != the probability of [0.01, 0.1] 
        # the probability of [log(0.001), log(0.01)] = the probability of [log(0.01), log(0.1)] 
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # clamp, make sure dt >= 1e-4
        dt = torch.clamp(dt, dt_init_floor)
        # inv softplus 
        # x = ln(e^y - 1) = y + ln(-(e^-y - 1)) to make sure numerical stability if dt is close to 0!
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # No weight decay. if weight decay, dt_bias is close to 0!
        self.dt_bias._no_weight_decay = True
        
        ## A parameter ##
        assert A_init_range[0] > 0 and A_init_range[1] > A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range) #[nheads]
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        ## D parameter ##
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
    def forward(self, u, seq_idx=None):
        """
        X: [B, L, D]
        A: [nheads]
        B: [B, L, ngroups * d_state]
        C: [B, L, ngroups * d_state]
        dt: [nheads] -> [B, L, nheads]
        """
        # u: (B, L, D)
        batch, seq_len, dim = u.shape
        
        # Step 1: Linear Projection
        # (B, L, D) -> (B, L, d_in_proj)
        zxbct = self.in_proj(u)
        A = -torch.exp(self.A_log) # [nheads]
        initial_states = repeat(self.initial_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path:
            pass
        else:
            # Step 2: 1D Convlution
            # [B, L, d_in_proj] -> [B, L, d_inner] [B, L, d_inner + 2 * ngroups * d_state] [B, L, nheads]
            residual, xBC, dt = torch.split(zxbct, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
            
            # softplus: y = ln(exp(x) + 1)
            # make sure dt > 0. A_bar = exp(dt * A), normaly A < 0, if dt < 0, then A_bar is > 1 ! It will lead to exponetial explosion
            # [nheads] -> [B, L, nheads] 
            dt = F.softplus(dt + self.dt_bias)
            assert self.activation in ["silu", "swish"]
            
            # Step 1: 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.conv1d(xBC.transpose(1, 2)) # [B, dim, L] -> [B, dim, L + K - 1]
                xBC = self.act(xBC.transpose(1, 2))
                xBC = xBC[:, :seq_len, :] # [B, L + K - 1, dim] -> [B, L, dim]
            else:
                # Pytorch : conv1d.weight: [out_channels, in_channels // groups, kernel_size]
                # CUDA : conv1d.weight: [out_channels, kernel_size]
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2).contiguous(),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
                
            # Step 2: Split into 3 main branches: X, B, C
            # Correspond to V, K, Q in attention 
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            # Step 3: Mamba scan
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            
            # Step 4: RMSNorm + residual
            y = self.norm(y, residual)
            # Step 5: Linear Projection
            out = self.out_proj(y)
        
        return out