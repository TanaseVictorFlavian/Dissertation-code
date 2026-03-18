import torch
import torch.nn as nn
import torch.nn.functional as F

class mLSTMCell(nn.Module):
    """
    mLSTM cell with matrix memory.
    Equations (16)-(18) from the paper.
    """
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q, k, v, i_tilde, f_tilde, state):
        # state: (C_prev, n_prev)
        # C_prev: (batch, head_dim, head_dim)
        # n_prev: (batch, head_dim)
        C_prev, n_prev = state
        
        # Exponential gating
        i_t = torch.exp(i_tilde) # (batch, 1)
        f_t = torch.exp(f_tilde) # (batch, 1)
        
        # (20) Key scaling factor
        k = k / (self.head_dim ** 0.5)
        
        # (16) Matrix cell state update
        # v: (batch, head_dim), k: (batch, head_dim)
        C_t = f_t.unsqueeze(-1) * C_prev + i_t.unsqueeze(-1) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        
        # (17) Normalizer state update
        n_t = f_t * n_prev + i_t * k
        
        # (18) Hidden state pre-activation
        # h_tilde = (C_t @ q) / max(|n_t.T @ q|, 1)
        dot_product = torch.sum(n_t * q, dim=-1, keepdim=True) # (batch, 1)
        h_tilde = torch.bmm(C_t, q.unsqueeze(2)).squeeze(2) / torch.max(torch.abs(dot_product), torch.ones_like(dot_product))
        
        return h_tilde, (C_t, n_t)

class mLSTMBlock(nn.Module):
    """
    mLSTM Block as shown in Figure 8.
    Pre up-projection residual structure.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Pre up-projection with projection factor PF=2
        self.PF = 2
        self.hidden_dim = dim * self.PF
        self.head_dim = self.hidden_dim // num_heads
        
        self.ln = nn.LayerNorm(dim)
        
        # Up-projection once for the externalized output gate and once for the mLSTM cells
        self.up_proj_gate = nn.Linear(dim, self.hidden_dim)
        self.up_proj_cell = nn.Linear(dim, self.hidden_dim)
        
        # Causal convolution (kernel size 4) applied dimension-wise
        self.conv1d = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=4, padding=3, groups=self.hidden_dim)
        
        # Block-diagonal Projections (implemented efficiently via Conv1d with groups)
        # q and k are projected block-diagonally from hidden_dim -> hidden_dim
        self.q_proj = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, groups=num_heads)
        self.k_proj = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, groups=num_heads)
        
        # i and f gates are scalar per head, projected block-diagonally from hidden_dim -> num_heads
        self.i_proj = nn.Conv1d(self.hidden_dim, num_heads, kernel_size=1, groups=num_heads)
        self.f_proj = nn.Conv1d(self.hidden_dim, num_heads, kernel_size=1, groups=num_heads)
        
        self.cells = nn.ModuleList([
            mLSTMCell(self.head_dim) for _ in range(num_heads)
        ])
        
        self.gn = nn.GroupNorm(num_heads, self.hidden_dim)
        
        # Learnable skip connection from the post-convolution input
        self.learnable_skip = nn.Parameter(torch.ones(self.hidden_dim))
        
        # Final down-projection
        self.down_proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        residual = x
        
        # 1. Pre LayerNorm
        x_norm = self.ln(x)
        
        # 2. Gate branch up-projection and activation (Swish)
        gate_pre = self.up_proj_gate(x_norm)
        gate_act = F.silu(gate_pre) # (batch, seq_len, hidden_dim)
        
        # 3. Cell branch up-projection
        cell_in = self.up_proj_cell(x_norm) # (batch, seq_len, hidden_dim)
        
        # 4. Values (v) skip the convolution and projections (fed directly)
        v = cell_in
        
        # 5. Causal Convolution Path
        conv_in = cell_in.transpose(1, 2) # (batch, hidden_dim, seq_len)
        conv_in = self.conv1d(conv_in)[:, :, :-3] # Remove padding to maintain causal property
        conv_in = F.silu(conv_in) # Swish activation
        
        # Block-diagonal projections for q, k, i, f
        q = self.q_proj(conv_in).transpose(1, 2) # (batch, seq_len, hidden_dim)
        k = self.k_proj(conv_in).transpose(1, 2)
        i_tilde = self.i_proj(conv_in).transpose(1, 2) # (batch, seq_len, num_heads)
        f_tilde = self.f_proj(conv_in).transpose(1, 2)
        
        # 6. mLSTM Cells (sequential over time)
        batch_size, seq_len, _ = x.shape
        
        q_heads = q.chunk(self.num_heads, dim=-1)
        k_heads = k.chunk(self.num_heads, dim=-1)
        v_heads = v.chunk(self.num_heads, dim=-1)
        
        head_outputs = []
        for h in range(self.num_heads):
            cell = self.cells[h]
            C_t = torch.zeros(batch_size, self.head_dim, self.head_dim, device=x.device)
            n_t = torch.zeros(batch_size, self.head_dim, device=x.device)
            state = (C_t, n_t)
            
            head_h = []
            for t in range(seq_len):
                h_t, state = cell(
                    q_heads[h][:, t, :], 
                    k_heads[h][:, t, :], 
                    v_heads[h][:, t, :], 
                    i_tilde[:, t, h:h+1], 
                    f_tilde[:, t, h:h+1], 
                    state
                )
                head_h.append(h_t.unsqueeze(1))
            
            head_outputs.append(torch.cat(head_h, dim=1))
            
        h_all = torch.cat(head_outputs, dim=-1)
        
        # 7. GroupNorm
        h_norm = h_all.transpose(1, 2)
        h_norm = self.gn(h_norm)
        h_norm = h_norm.transpose(1, 2)
        
        # 8. Learnable Skip Connection and Gating
        skip = conv_in.transpose(1, 2) * self.learnable_skip
        h_combined = h_norm + skip
        h_gated = h_combined * gate_act
        
        # 9. Down-projection and residual connection
        out = self.down_proj(h_gated)
        
        return residual + out
