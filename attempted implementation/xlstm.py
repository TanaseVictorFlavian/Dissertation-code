"""
xLSTM: Extended Long Short-Term Memory
=======================================
A clean, self-contained PyTorch implementation following:
  Beck et al., "xLSTM: Extended Long Short-Term Memory", NeurIPS 2024
  HuggingFace Transformers v5.4.0 reference implementation

Architecture (bottom-up):
  1. sLSTMCell  — Scalar memory with exponential gating (Eq. 49-53)
  2. mLSTMCell  — Matrix (covariance) memory with exponential gating (Eq. 16-18)
  3. sLSTMBlock — Pre-norm residual block with sLSTM + gated MLP (Fig. 7)
  4. mLSTMBlock — Pre-norm residual block with mLSTM + up/down projection (Fig. 8)
  5. xLSTM      — Full architecture: Embedding → [sLSTM/mLSTM blocks] → Norm → LM Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class sLSTMCell(nn.Module):
    """
    sLSTM cell — scalar cell state with exponential input/forget gates.

    State: (h, c, n, m) where
      h: hidden state          (batch, head_dim)
      c: cell state            (batch, head_dim)
      n: normalizer state      (batch, head_dim)
      m: stabilizer state      (batch, head_dim)
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

        # Recurrent weight matrices for memory mixing (block-diagonal globally)
        self.R_z = nn.Linear(head_dim, head_dim, bias=False)  # cell input
        self.R_i = nn.Linear(head_dim, head_dim, bias=False)  # input gate
        self.R_f = nn.Linear(head_dim, head_dim, bias=False)  # forget gate
        self.R_o = nn.Linear(head_dim, head_dim, bias=False)  # output gate

    def forward(
        self,
        z_pre: torch.Tensor,   # cell input pre-activation  (batch, head_dim)
        i_pre: torch.Tensor,   # input gate pre-activation  (batch, head_dim)
        f_pre: torch.Tensor,   # forget gate pre-activation (batch, head_dim)
        o_pre: torch.Tensor,   # output gate pre-activation (batch, head_dim)
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        h_prev, c_prev, n_prev, m_prev = state

        # Add recurrent connections (hidden-to-hidden)
        z_tilde = z_pre + self.R_z(h_prev)
        i_tilde = i_pre + self.R_i(h_prev)
        f_tilde = f_pre + self.R_f(h_prev)
        o_tilde = o_pre + self.R_o(h_prev)

        # Eq. (34): Cell input activation
        z_t = torch.tanh(z_tilde)

        # Eq. (37): Output gate (standard sigmoid, not exponential)
        o_t = torch.sigmoid(o_tilde)

        # Eq. (49): Stabilizer state update
        #   m_t = max(f̃_t + m_{t-1}, ĩ_t)
        m_t = torch.max(f_tilde + m_prev, i_tilde)

        # Eq. (50): Stabilized input gate
        #   i'_t = exp(ĩ_t - m_t)
        i_prime = torch.exp(i_tilde - m_t)

        # Eq. (51): Stabilized forget gate
        #   f'_t = exp(f̃_t + m_{t-1} - m_t)
        f_prime = torch.exp(f_tilde + m_prev - m_t)

        # Eq. (52): Cell state update
        #   c_t = f'_t * c_{t-1} + i'_t * z_t
        c_t = f_prime * c_prev + i_prime * z_t

        # Eq. (53): Normalizer state update
        #   n_t = f'_t * n_{t-1} + i'_t
        n_t = f_prime * n_prev + i_prime

        # Eq. (33): Hidden state
        #   h_t = o_t * (c_t / n_t)
        h_t = o_t * (c_t / torch.max(n_t, torch.ones_like(n_t)))

        return h_t, (h_t, c_t, n_t, m_t)


# =============================================================================
# Section 2: mLSTM Cell  (Matrix memory, exponential gating)
# Paper Equations: (16)-(18), (20)
# =============================================================================

class mLSTMCell(nn.Module):
    """
    mLSTM cell — matrix (covariance) cell state with exponential gating.

    Key innovation over sLSTM:
      - Cell state C is a matrix (d x d), not a scalar vector
      - Uses query/key/value projections (like attention)
      - Fully parallelizable (no hidden-to-hidden recurrence)
      - Memory capacity scales quadratically with dimension

    State: (C, n) where
      C: matrix cell state     (batch, head_dim, head_dim)
      n: normalizer state      (batch, head_dim)
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self,
        q: torch.Tensor,       # query            (batch, head_dim)
        k: torch.Tensor,       # key              (batch, head_dim)
        v: torch.Tensor,       # value            (batch, head_dim)
        i_tilde: torch.Tensor, # input gate pre   (batch, 1)
        f_tilde: torch.Tensor, # forget gate pre  (batch, 1)
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        C_prev, n_prev, m_prev = state

        # Stabilizer state update
        m_t = torch.max(f_tilde + m_prev, i_tilde)

        # Exponential gates (scalar per head)
        i_prime = torch.exp(i_tilde - m_t)  # (batch, 1)
        f_prime = torch.exp(f_tilde + m_prev - m_t)  # (batch, 1)

        # Eq. (20): Scale key by 1/√d for stable dot products
        k_scaled = k / math.sqrt(self.head_dim)

        # Eq. (16): Matrix cell state update
        v_outer_k = torch.bmm(v.unsqueeze(2), k_scaled.unsqueeze(1))
        C_t = f_prime.unsqueeze(-1) * C_prev + i_prime.unsqueeze(-1) * v_outer_k

        # Eq. (17): Normalizer state update
        n_t = f_prime * n_prev + i_prime * k_scaled

        # Eq. (18): Hidden state computation
        #   h̃_t = (C_t @ q) / max(|n_t^T q|, 1)
        # We must also scale the bound by exp(-m_t) because C_t and n_t are natively stabilized by m_t
        numerator = torch.bmm(C_t, q.unsqueeze(2)).squeeze(2)  # (batch, head_dim)
        denominator = torch.sum(n_t * q, dim=-1, keepdim=True)  # (batch, 1)
        denominator = torch.max(torch.abs(denominator), torch.exp(-m_t))
        h_tilde = numerator / denominator

        return h_tilde, (C_t, n_t, m_t)


# =============================================================================
# Section 3: sLSTM Block (Figure 7 of the paper)
# Post up-projection residual structure
# =============================================================================

class sLSTMBlock(nn.Module):
    """
    sLSTM Block (Figure 7):

    """

    def __init__(self, dim: int, num_heads: int = 4, use_conv1d: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_conv1d = use_conv1d

        # Pre-norm
        self.ln1 = nn.LayerNorm(dim)

        if self.use_conv1d:
            # Causal convolution for input/forget gates only (kernel=4)
            self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, groups=dim, padding=3)

        # Block-diagonal projections via grouped Conv1d
        # i, f gates get convolved input
        self.i_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        self.f_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        # z (cell input), o (output gate) get raw input
        self.z_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)
        self.o_proj = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads)

        # Multi-head sLSTM cells
        self.cells = nn.ModuleList([
            sLSTMCell(self.head_dim) for _ in range(num_heads)
        ])

        # Group normalization (one group per head)
        self.gn = nn.GroupNorm(num_heads, dim)

        # Gated MLP (post up-projection, projection factor ≈ 4/3)
        self.ln2 = nn.LayerNorm(dim)
        pf_dim = int(dim * 4 / 3)
        self.up_proj = nn.Linear(dim, pf_dim)
        self.gate_proj = nn.Linear(dim, pf_dim)
        self.down_proj = nn.Linear(pf_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            out: (batch, seq_len, dim) — same shape as input
        """
        residual = x
        batch_size, seq_len, _ = x.shape

        # 1. Pre-LayerNorm
        x_norm = self.ln1(x)
        x_norm_t = x_norm.transpose(1, 2)  # (batch, dim, seq_len)

        # 2. Causal convolution path (for input and forget gates)
        if self.use_conv1d:
            x_conv = self.conv1d(x_norm_t)[:, :, :seq_len]  # causal: trim future
            x_conv = F.silu(x_conv)
        else:
            x_conv = x_norm_t

        # 3. Block-diagonal projections
        i_pre = self.i_proj(x_conv).transpose(1, 2)   # (batch, seq_len, dim)
        f_pre = self.f_proj(x_conv).transpose(1, 2)
        z_pre = self.z_proj(x_norm_t).transpose(1, 2)  # bypass conv
        o_pre = self.o_proj(x_norm_t).transpose(1, 2)  # bypass conv

        # 4. Multi-head sLSTM (sequential over time)
        i_heads = i_pre.chunk(self.num_heads, dim=-1)
        f_heads = f_pre.chunk(self.num_heads, dim=-1)
        z_heads = z_pre.chunk(self.num_heads, dim=-1)
        o_heads = o_pre.chunk(self.num_heads, dim=-1)

        head_outputs = []
        for h_idx in range(self.num_heads):
            cell = self.cells[h_idx]
            # Initialize state
            h_t = torch.zeros(batch_size, self.head_dim, device=x.device, dtype=x.dtype)
            c_t = torch.zeros(batch_size, self.head_dim, device=x.device, dtype=x.dtype)
            n_t = torch.ones(batch_size, self.head_dim, device=x.device, dtype=x.dtype)
            m_t = torch.zeros(batch_size, self.head_dim, device=x.device, dtype=x.dtype)
            state = (h_t, c_t, n_t, m_t)

            timestep_outputs = []
            for t in range(seq_len):
                h_t, state = cell(
                    z_heads[h_idx][:, t, :],
                    i_heads[h_idx][:, t, :],
                    f_heads[h_idx][:, t, :],
                    o_heads[h_idx][:, t, :],
                    state,
                )
                timestep_outputs.append(h_t.unsqueeze(1))

            head_outputs.append(torch.cat(timestep_outputs, dim=1))

        h_out = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, dim)

        # 5. GroupNorm 
        # h_norm = (B, T, D) -> (B, D, T) -> (B, T, D)
        h_norm = self.gn(h_out.transpose(1, 2)).transpose(1, 2)

        # 6. Residual connection
        x = residual + h_norm

        # 7. Gated MLP (post up-projection)
        mlp_residual = x
        x_ln = self.ln2(x)
        gate = F.gelu(self.gate_proj(x_ln))
        up = self.up_proj(x_ln)
        x = self.down_proj(up * gate)

        return mlp_residual + x

class mLSTMBlock(nn.Module):
    """
    mLSTM Block (Figure 8):

    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Pre up-projection factor
        self.PF = 2
        self.hidden_dim = dim * self.PF
        self.head_dim = self.hidden_dim // num_heads

        # Pre-norm
        self.ln = nn.LayerNorm(dim)

        # Dual up-projections
        self.up_proj_gate = nn.Linear(dim, self.hidden_dim)  # gate branch
        self.up_proj_cell = nn.Linear(dim, self.hidden_dim)  # cell branch

        # Causal convolution (kernel=4, dimension-wise)
        self.conv1d = nn.Conv1d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=4, padding=3, groups=self.hidden_dim,
        )

        # Block-diagonal projections for q, k (grouped conv1d)
        self.q_proj = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, groups=num_heads)
        self.k_proj = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, groups=num_heads)

        # Scalar gates per head (i, f)
        self.i_proj = nn.Conv1d(self.hidden_dim, num_heads, kernel_size=1, groups=num_heads)
        self.f_proj = nn.Conv1d(self.hidden_dim, num_heads, kernel_size=1, groups=num_heads)

        # Multi-head mLSTM cells
        self.cells = nn.ModuleList([
            mLSTMCell(self.head_dim) for _ in range(num_heads)
        ])

        # Group normalization
        self.gn = nn.GroupNorm(num_heads, self.hidden_dim)

        # Learnable skip connection
        self.learnable_skip = nn.Parameter(torch.ones(self.hidden_dim))

        # Down-projection back to model dimension
        self.down_proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            out: (batch, seq_len, dim) — same shape as input
        """
        residual = x
        batch_size, seq_len, _ = x.shape

        # 1. Pre-LayerNorm
        x_norm = self.ln(x)

        # 2. Gate branch: up-project + Swish
        gate = F.silu(self.up_proj_gate(x_norm))  # (batch, seq_len, hidden_dim)

        # 3. Cell branch: up-project
        cell_in = self.up_proj_cell(x_norm)  # (batch, seq_len, hidden_dim)

        # 4. Values skip convolution (fed directly to mLSTM)
        v = cell_in  # (batch, seq_len, hidden_dim)

        # 5. Causal convolution path
        conv_in = cell_in.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        conv_out = self.conv1d(conv_in)[:, :, :seq_len]  # causal trim
        conv_out = F.silu(conv_out)

        # 6. Block-diagonal projections
        q = self.q_proj(conv_out).transpose(1, 2)  # (batch, seq_len, hidden_dim)
        k = self.k_proj(conv_out).transpose(1, 2)
        i_tilde = self.i_proj(conv_out).transpose(1, 2)  # (batch, seq_len, num_heads)
        f_tilde = self.f_proj(conv_out).transpose(1, 2)

        # 7. Multi-head mLSTM (sequential over time)
        q_heads = q.chunk(self.num_heads, dim=-1)
        k_heads = k.chunk(self.num_heads, dim=-1)
        v_heads = v.chunk(self.num_heads, dim=-1)

        head_outputs = []
        for h_idx in range(self.num_heads):
            cell = self.cells[h_idx]
            C_t = torch.zeros(batch_size, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
            n_t = torch.zeros(batch_size, self.head_dim, device=x.device, dtype=x.dtype)
            m_t = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            state = (C_t, n_t, m_t)

            timestep_outputs = []
            for t in range(seq_len):
                h_t, state = cell(
                    q_heads[h_idx][:, t, :],
                    k_heads[h_idx][:, t, :],
                    v_heads[h_idx][:, t, :],
                    i_tilde[:, t, h_idx:h_idx + 1],
                    f_tilde[:, t, h_idx:h_idx + 1],
                    state,
                )
                timestep_outputs.append(h_t.unsqueeze(1))

            head_outputs.append(torch.cat(timestep_outputs, dim=1))

        h_all = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, hidden_dim)

        # 8. GroupNorm
        h_norm = self.gn(h_all.transpose(1, 2)).transpose(1, 2)

        # 9. Learnable skip connection
        skip = conv_out.transpose(1, 2) * self.learnable_skip
        h_combined = h_norm + skip

        # 10. Output gating and down-projection
        h_gated = h_combined * gate
        out = self.down_proj(h_gated)

        return residual + out


# =============================================================================
# Section 5: xLSTM Architecture (Full model)
# Embedding → [sLSTM/mLSTM blocks interleaved] → RMSNorm → LM Head
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class xLSTMConfig:
    """Configuration for the xLSTM architecture."""

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        block_types: Optional[List[str]] = None,
        norm_eps: float = 1e-6,
        tie_embeddings: bool = False,
        slstm_use_conv1d: bool = True,
        slstm_normal_init: bool = False,
    ):
        """
        Args:
            vocab_size: Size of the token vocabulary.
            dim: Model hidden dimension.
            num_layers: Number of xLSTM blocks.
            num_heads: Number of attention/memory heads per block.
            block_types: Pattern of block types, e.g. ["m","m","s","m","m","s"].
                         "s" = sLSTM block, "m" = mLSTM block.
                         If None, defaults to alternating m/s pattern.
            norm_eps: Epsilon for normalization layers.
            tie_embeddings: Whether to tie input/output embeddings.
        """
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.slstm_use_conv1d = slstm_use_conv1d
        self.slstm_normal_init = slstm_normal_init

        if block_types is None:
            # Default: mostly mLSTM with periodic sLSTM (ratio 7:1 in the paper)
            # For small models, alternate m and s
            self.block_types = []
            for i in range(num_layers):
                if (i + 1) % 3 == 0:  # every 3rd block is sLSTM
                    self.block_types.append("s")
                else:
                    self.block_types.append("m")
        else:
            assert len(block_types) == num_layers
            self.block_types = block_types


class xLSTM(nn.Module):
    """
    xLSTM: Extended Long Short-Term Memory

    Full architecture combining sLSTM and mLSTM blocks:
      - Token embedding
      - Stack of residual blocks (sLSTM and/or mLSTM)
      - RMS normalization
      - Language model head (linear projection to vocabulary)

    The block pattern is configurable. The paper recommends a 7:1 ratio
    of mLSTM to sLSTM blocks, e.g. [m,m,m,m,m,m,m,s,...].

    Example usage:
        config = xLSTMConfig(vocab_size=32000, dim=128, num_layers=6, num_heads=4)
        model = xLSTM(config)
        input_ids = torch.randint(0, 32000, (2, 64))
        logits = model(input_ids)  # (2, 64, 32000)
    """

    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # Build block stack based on config pattern
        self.blocks = nn.ModuleList()
        for block_type in config.block_types:
            if block_type == "s":
                self.blocks.append(sLSTMBlock(config.dim, config.num_heads, use_conv1d=config.slstm_use_conv1d))
            elif block_type == "m":
                self.blocks.append(mLSTMBlock(config.dim, config.num_heads))
            else:
                raise ValueError(f"Unknown block type: {block_type}. Use 's' or 'm'.")

        # Output normalization
        self.out_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Optionally tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Apply weight initialization following the paper's recommendations."""
        for name, module in self.named_modules():
            # If the user requested normal init for slstm and this module is inside an sLSTMBlock
            if getattr(self.config, "slstm_normal_init", False) and "blocks." in name:
                parts = name.split(".")
                try:
                    block_idx = int(parts[1])
                    if isinstance(self.blocks[block_idx], sLSTMBlock):
                        continue  # Skip custom small init, use PyTorch default (normal init)
                except (IndexError, ValueError):
                    pass

            if isinstance(module, nn.Linear):
                # Small init for most linear layers
                std = (2 / (5 * module.weight.shape[1])) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                std = (2 / (5 * module.weight.shape[1])) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass of the xLSTM language model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).

        Returns:
            logits: Prediction scores of shape (batch_size, seq_len, vocab_size).
        """
        # Token embedding
        x = self.embedding(input_ids)  # (batch, seq_len, dim)

        # Pass through block stack
        for block in self.blocks:
            x = block(x)

        # Output normalization
        x = self.out_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count total (or trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Quick test / demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("xLSTM Implementation Test")
    print("=" * 60)

    # --- Test individual cells ---
    print("\n1. Testing sLSTMCell...")
    cell_s = sLSTMCell(head_dim=16)
    h = torch.zeros(2, 16)
    c = torch.zeros(2, 16)
    n = torch.ones(2, 16)
    m = torch.zeros(2, 16)
    state_s = (h, c, n, m)
    z_pre = torch.randn(2, 16)
    i_pre = torch.randn(2, 16)
    f_pre = torch.randn(2, 16)
    o_pre = torch.randn(2, 16)
    h_out, state_s = cell_s(z_pre, i_pre, f_pre, o_pre, state_s)
    print(f"   h_out shape: {h_out.shape}")  # (2, 16)
    print("   ✓ sLSTMCell OK")

    print("\n2. Testing mLSTMCell...")
    cell_m = mLSTMCell(head_dim=16)
    C = torch.zeros(2, 16, 16)
    n_m = torch.zeros(2, 16)
    m_m = torch.zeros(2, 1)
    state_m = (C, n_m, m_m)
    q = torch.randn(2, 16)
    k = torch.randn(2, 16)
    v = torch.randn(2, 16)
    i_t = torch.randn(2, 1)
    f_t = torch.randn(2, 1)
    h_out, state_m = cell_m(q, k, v, i_t, f_t, state_m)
    print(f"   h_out shape: {h_out.shape}")  # (2, 16)
    print("   ✓ mLSTMCell OK")

    # --- Test blocks ---
    print("\n3. Testing sLSTMBlock...")
    block_s = sLSTMBlock(dim=32, num_heads=4)
    x = torch.randn(2, 8, 32)
    out = block_s(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ sLSTMBlock OK")

    print("\n4. Testing mLSTMBlock...")
    block_m = mLSTMBlock(dim=32, num_heads=4)
    out = block_m(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ mLSTMBlock OK")

    # --- Test full model ---
    print("\n5. Testing full xLSTM model...")
    config = xLSTMConfig(
        vocab_size=1000,
        dim=64,
        num_layers=6,
        num_heads=4,
        block_types=["m", "m", "s", "m", "m", "s"],
    )
    model = xLSTM(config)
    input_ids = torch.randint(0, 1000, (2, 16))
    logits = model(input_ids)
    print(f"   Input:  {input_ids.shape}")
    print(f"   Output: {logits.shape}")
    assert logits.shape == (2, 16, 1000)

    n_params = model.num_parameters()
    print(f"   Parameters: {n_params:,}")
    print("   ✓ xLSTM OK")

    print(f"\n   Block pattern: {config.block_types}")
    for i, block in enumerate(model.blocks):
        btype = "sLSTM" if isinstance(block, sLSTMBlock) else "mLSTM"
        print(f"   Block {i}: {btype}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
