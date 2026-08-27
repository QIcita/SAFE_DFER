"""Spatio-Temporal Modeling Module (STM) with S2AM2."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utility
# =========================================================
def create_sliding_windows(x, window_size=4, step_size=2):
    """
    x: [B, C, T, H, W]
    return: [B, Nw, C, window_size, H, W]
    """
    b, c, t, h, w = x.shape
    windows = []
    for i in range(0, t - window_size + 1, step_size):
        windows.append(x[:, :, i : i + window_size, :, :])
    return torch.stack(windows, dim=1)


# =========================================================
# Lightweight Self-Attention Block
# =========================================================
class FullAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # q, k, v: [B, heads, T, dim]
        dim = q.size(-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (dim**0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = FullAttention(dropout=dropout)

    def forward(self, x):
        # x: [B, T, D]
        b, t, d = x.shape
        q = (
            self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        )  # [B,H,T,Dh]
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        out, attn = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        return out, attn


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = MultiHeadSelfAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_map = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_map


# =========================================================
# STM Core: S2AM2-style Cell
# =========================================================
class S2AM2Cell(nn.Module):
    """
    Implements the three core STM mechanisms described in SAFE:
    1) sparse gating;
    2) state-aware scaling;
    3) residual state fusion.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM-like gates
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

        # 1) Sparse gating
        self.sparse_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 2) State-aware scaling
        self.state_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 3) Residual state fusion
        self.res_gate = nn.Linear(hidden_dim, hidden_dim)
        self.res_value = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, h_prev, c_prev):
        # standard recurrent update
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))
        c_hat_t = torch.tanh(self.W_c(x_t) + self.U_c(h_prev))

        # -------------------------------------------------
        # (1) Sparse gating suppresses uninformative temporal states.
        # M_t = sigma(MLP(h_{t-1})) produces a feature-wise gate.
        # -------------------------------------------------
        m_t = self.sparse_gate(h_prev)  # [B, H]
        c_hat_t = c_hat_t * m_t

        # standard cell state
        c_t = f_t * c_prev + i_t * c_hat_t

        # -------------------------------------------------
        # (2) State-aware scaling
        # Adaptively scale the current state to emphasize useful dimensions.
        # -------------------------------------------------
        w_a = self.state_scale(c_t)
        c_t = c_t * w_a

        # -------------------------------------------------
        # (3) Residual state fusion
        # Inject residual history to stabilize long-range information flow.
        # -------------------------------------------------
        res = torch.sigmoid(self.res_gate(h_prev)) * F.relu(self.res_value(h_prev))
        c_t = c_t + self.dropout(res)

        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t, m_t, w_a


class S2AM2Layer(nn.Module):
    """
    Unidirectional temporal modeling layer.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                S2AM2Cell(
                    input_dim if i == 0 else hidden_dim, hidden_dim, dropout=dropout
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: [B, T, D]
        b, t, _ = x.shape
        h = [
            torch.zeros(b, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(b, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []
        sparse_gates = []
        state_scales = []

        for step in range(t):
            x_t = x[:, step, :]
            for layer_idx, layer in enumerate(self.layers):
                h[layer_idx], c[layer_idx], m_t, w_a = layer(
                    x_t, h[layer_idx], c[layer_idx]
                )
                x_t = h[layer_idx]
            outputs.append(h[-1])
            sparse_gates.append(m_t)
            state_scales.append(w_a)

        outputs = torch.stack(outputs, dim=1)  # [B, T, H]
        sparse_gates = torch.stack(sparse_gates, dim=1)  # [B, T, H]
        state_scales = torch.stack(state_scales, dim=1)  # [B, T, H]
        return outputs, sparse_gates, state_scales


class BiS2AM2Layer(nn.Module):
    """
    Bidirectional temporal modeling with S2AM2 cells.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.fwd = S2AM2Layer(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.bwd = S2AM2Layer(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout
        )

    def forward(self, x):
        # x: [B, T, D]
        out_fwd, gate_fwd, scale_fwd = self.fwd(x)

        x_rev = torch.flip(x, dims=[1])
        out_bwd, gate_bwd, scale_bwd = self.bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        gate_bwd = torch.flip(gate_bwd, dims=[1])
        scale_bwd = torch.flip(scale_bwd, dims=[1])

        out = torch.cat([out_fwd, out_bwd], dim=-1)  # [B, T, 2H]
        gate = torch.cat([gate_fwd, gate_bwd], dim=-1)
        scale = torch.cat([scale_fwd, scale_bwd], dim=-1)
        return out, gate, scale


# =========================================================
# Temporal Attention Pooling
# Lightweight temporal reweighting and aggregation.
# =========================================================
class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, D]
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)  # [B, T]
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, attn


# =========================================================
# SAFE STM Module
# =========================================================
class SAFE_STM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, dropout=0.1, n_heads=8):
        super().__init__()

        # Stage 1: unidirectional temporal modeling.
        self.stage1 = S2AM2Layer(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, dropout=dropout
        )

        # Intermediate lightweight self-attention.
        self.attn1 = AttentionBlock(
            d_model=hidden_dim, n_heads=n_heads, dropout=dropout
        )

        # Stage 2: bidirectional temporal modeling.
        self.stage2 = BiS2AM2Layer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=1, dropout=dropout
        )

        # Second lightweight self-attention block.
        self.attn2 = AttentionBlock(
            d_model=hidden_dim * 2, n_heads=n_heads, dropout=dropout
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)

        # Combine attention pooling with temporal convolution.
        self.temporal_pool = TemporalAttentionPooling(hidden_dim * 2)
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=1,
        )

    def forward(self, x):
        """
        x: [B, T, 512]
        return:
            seq_feat: [B, T, 1024]
            video_feat: [B, 1024]
            aux: dict
        """
        out1, gate1, scale1 = self.stage1(x)  # [B,T,512]
        out1, attn_map1 = self.attn1(out1)

        out2, gate2, scale2 = self.stage2(out1)  # [B,T,1024]
        out2, attn_map2 = self.attn2(out2)

        seq_feat = self.norm(out2)

        # attention pooling
        pooled_attn, temporal_attn = self.temporal_pool(seq_feat)

        # temporal conv pooling
        conv_feat = self.temporal_conv(seq_feat.transpose(1, 2)).mean(dim=-1)

        # Fuse the two temporal aggregation paths.
        video_feat = pooled_attn + conv_feat

        aux = {
            "gate_stage1": gate1,
            "scale_stage1": scale1,
            "gate_stage2": gate2,
            "scale_stage2": scale2,
            "attn_map1": attn_map1,
            "attn_map2": attn_map2,
            "temporal_attn": temporal_attn,
        }

        return seq_feat, video_feat, aux


# Paper-facing module name.
STM = SAFE_STM

__all__ = [
    "create_sliding_windows",
    "S2AM2Cell",
    "S2AM2Layer",
    "BiS2AM2Layer",
    "SAFE_STM",
    "STM",
]
