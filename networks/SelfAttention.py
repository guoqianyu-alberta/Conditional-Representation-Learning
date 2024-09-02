import torch
from torch import nn
import torch.nn.functional as F
from .position_embedding import PositionEmbeddingSine


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class SelfAttentionLayer(nn.Module):
    def __init__(
            self,
            input_channel,
            d_model,
            nhead,
            dropout=0.0
    ):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.hidden_dim = d_model
        self.pre_norm = nn.LayerNorm(d_model)
        self.after_norm = nn.LayerNorm(d_model)
        self.input_channel = input_channel
        self.ffn_layer = FFNLayer(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.multihead_attn(query=self.pre_norm(x),
                                     key=self.pre_norm(x),
                                     value=self.pre_norm(x))[0]
        # Add & Norm
        output = self.after_norm(output + x)
        # FFn + Add & Norm
        output = self.ffn_layer(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_channel, hidden_dim, num_heads=8, dropout=0.0):
        super(SelfAttention, self).__init__()
        n_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(n_steps, normalize=True)
        self.self_encoder = SelfAttentionLayer(input_channel, hidden_dim, num_heads, dropout)
        self.hidden_dim = hidden_dim
        self.s_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.q_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.mode = "sharing_weight"

    def attn_input_process(self, x, proj_function, pos=None):
        x = proj_function(x).flatten(-2).permute(2, 0, 1) # [h*w, bsz, hidden_dim]
        if pos is not None:
            pos = pos.flatten(-2).permute(2, 0, 1)
            return x + pos
        return x

    def forward(self, support_feat, query_feat):
        s_pos = self.pe_layer(support_feat, None)
        q_pos = self.pe_layer(query_feat, None)
        support_feat = self.attn_input_process(support_feat, self.s_proj, s_pos)
        query_feat = self.attn_input_process(query_feat, self.q_proj, q_pos)
        
        _, s_bsz, _ = support_feat.shape
        feat = torch.concat((support_feat, query_feat), dim=1)
        feat = self.self_encoder(feat).permute(1, 2, 0) # [bsz, hidden_dim, h*w]

        support_feat = feat[:s_bsz]
        query_feat = feat[s_bsz:]

        return support_feat, query_feat





