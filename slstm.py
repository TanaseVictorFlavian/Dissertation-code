import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTMCell(nn.Module):
    """
    sLSTM cell with stabilized exponential gating and recurrent connections.
    Equations (49)-(53) from the paper.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Recurrent connections for memory mixing within the head
        # These are intrinsically block-diagonal globally because they are instantiated per-head
        self.R_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, z_pre, o_pre, i_pre, f_pre, state):
        # state: (h, c, n, m)
        h_prev, c_prev, n_prev, m_prev = state
        
        # Add recurrent hidden-hidden connections
        i_tilde = i_pre + self.R_i(h_prev)
        f_tilde = f_pre + self.R_f(h_prev)
        z_tilde = z_pre + self.R_z(h_prev)
        o_tilde = o_pre + self.R_o(h_prev)
        
        # (34) cell input
        z_t = torch.tanh(z_tilde)
        
        # (37) output gate
        o_t = torch.sigmoid(o_tilde)
        
        # (49) stabilizer state
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        
        # (50) stabilized input gate
        i_s = torch.exp(i_tilde - m_t)
        
        # (51) stabilized forget gate
        f_s = torch.exp(f_tilde + m_prev - m_t)
        
        # (52) stabilized cell state update
        c_t = f_s * c_prev + i_s * z_t
        
        # (53) stabilized normalizer state update
        n_t = f_s * n_prev + i_s
        
        # (33) hidden state
        h_t = o_t * (c_t / n_t)
        
        return h_t, (h_t, c_t, n_t, m_t)

class sLSTMBlock(nn.Module):
    """
    sLSTM Block as shown in Figure 7.
    Post up-projection residual structure.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.ln1 = nn.LayerNorm(dim)
        
        # Causal convolution only for input and forget gates (Figure 7 & Caption)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, groups=dim, padding=3)
        
        # Block-diagonal Projections (implemented efficiently via Conv1d with groups)
        # i, f get convoluted input
        self.i_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        self.f_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        
        # z, o get raw LayerNorm input
        self.z_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        self.o_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        
        # Multi-head sLSTM
        self.cells = nn.ModuleList([
            sLSTMCell(self.head_dim) for _ in range(num_heads)
        ])
        
        self.gn = nn.GroupNorm(num_heads, dim)
        
        # Gated MLP (Post up-projection, PF=4/3)
        self.ln2 = nn.LayerNorm(dim)
        pf_dim = int(dim * 4/3)
        self.up_proj = nn.Linear(dim, pf_dim)
        self.gate_proj = nn.Linear(dim, pf_dim)
        self.down_proj = nn.Linear(pf_dim, dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        residual = x
        
        # 1. LayerNorm
        x_norm = self.ln1(x)
        x_norm_t = x_norm.transpose(1, 2) # (batch, dim, seq_len)
        
        # 2. Causal Convolution Path (only for i, f gates)
        x_conv = self.conv1d(x_norm_t)[:, :, :-3] # Remove padding for causal convolution
        x_conv = F.silu(x_conv) # Swish activation
        
        # 3. Block-diagonal Projections
        i_pre = self.i_proj(x_conv).transpose(1, 2) # (batch, seq_len, dim)
        f_pre = self.f_proj(x_conv).transpose(1, 2)
        z_pre = self.z_proj(x_norm_t).transpose(1, 2) # Bypasses convolution
        o_pre = self.o_proj(x_norm_t).transpose(1, 2) # Bypasses convolution
        
        # 4. sLSTM Cells (sequential over time)
        batch_size, seq_len, _ = x.shape
        
        # Split into heads
        i_pre_heads = i_pre.chunk(self.num_heads, dim=-1)
        f_pre_heads = f_pre.chunk(self.num_heads, dim=-1)
        z_pre_heads = z_pre.chunk(self.num_heads, dim=-1)
        o_pre_heads = o_pre.chunk(self.num_heads, dim=-1)
        
        head_outputs = []
        for h in range(self.num_heads):
            cell = self.cells[h]
            h_t = torch.zeros(batch_size, self.head_dim, device=x.device)
            c_t = torch.zeros(batch_size, self.head_dim, device=x.device)
            n_t = torch.ones(batch_size, self.head_dim, device=x.device)
            m_t = torch.zeros(batch_size, self.head_dim, device=x.device)
            state = (h_t, c_t, n_t, m_t)
            
            head_h = []
            for t in range(seq_len): 
                h_t, state = cell(
                    z_pre_heads[h][:, t, :],
                    o_pre_heads[h][:, t, :],
                    i_pre_heads[h][:, t, :],
                    f_pre_heads[h][:, t, :],
                    state
                )
                head_h.append(h_t.unsqueeze(1))
            
            head_outputs.append(torch.cat(head_h, dim=1))
            
        h_out = torch.cat(head_outputs, dim=-1)
        
        # 5. GroupNorm
        h_norm = h_out.transpose(1, 2)
        h_norm = self.gn(h_norm)
        h_norm = h_norm.transpose(1, 2)
        
        # 6. Residual connection
        x = residual + h_norm
        
        # 7. Gated MLP (Post up-projection)
        mlp_res = x
        x_ln2 = self.ln2(x)
        
        up = self.up_proj(x_ln2)
        gate = F.gelu(self.gate_proj(x_ln2))
        mlp_out = self.down_proj(up * gate)
        
        return mlp_res + mlp_out
