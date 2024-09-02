import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .position_embedding import PositionEmbeddingSine
import fvcore.nn.weight_init as weight_init
from torch.nn import TransformerDecoderLayer


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


# cross-attention
class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1
    ):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.hidden_dim = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # self.activation = nn.ReLU()
        self.ffn_layer = FFNLayer(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value):
        output = self.multihead_attn(query=query,
                                     key=key,
                                     value=value)[0]
        output = self.norm(query + self.dropout(output))
        output = self.ffn_layer(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, input_channel, hidden_dim, num_heads=8, dropout=0.0):
        super().__init__()
        n_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(n_steps, normalize=True)
        self.mode = "sharing_weight"
        self.cross_attention = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        self.hidden_dim = hidden_dim
        self.s_q_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.s_k_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.s_v_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.q_q_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.q_k_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.q_v_proj = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attn_input_process(self, raw_input, proj_function, pos=None):
        x = raw_input
        x = proj_function(x).flatten(-2).permute(2, 0, 1) # [sequence_length, bsz, channel]
        if pos is not None:
            pos = pos.flatten(-2).permute(2, 0, 1)
            return x + pos
        return x

    def forward(self, support_feat, query_feat):
        bsz_s, _, h, w = support_feat.shape
        bsz_q, _, h, w = query_feat.shape
        concat_key = torch.concat((support_feat, query_feat), dim=2)
        concat_value = torch.concat((support_feat, query_feat), dim=2)

        key_pos = self.pe_layer(concat_key, None)
        support_key = self.attn_input_process(concat_key.clone(), self.s_k_proj, key_pos.clone())
        query_key = self.attn_input_process(concat_key.clone(), self.q_k_proj, key_pos.clone())

        # value does not need position encoding
        support_value = self.attn_input_process(concat_value.clone(), self.s_v_proj, None)
        query_value = self.attn_input_process(concat_value.clone(), self.q_v_proj, None)

        query_pos = self.pe_layer(support_feat.clone())
        support_query = self.attn_input_process(support_feat.clone(), self.s_q_proj, query_pos.clone())
        query_query = self.attn_input_process(query_feat.clone(), self.q_q_proj, query_pos.clone())

        after_attn_s_feat = self.cross_attention(support_query, support_key, support_value)
        after_attn_q_feat = self.cross_attention(query_query, query_key, query_value)

        after_attn_s_feat = after_attn_s_feat.permute(1, 2, 0).view(bsz_s, self.hidden_dim, h, w)
        after_attn_q_feat = after_attn_q_feat.permute(1, 2, 0).view(bsz_q, self.hidden_dim, h, w)

        return after_attn_s_feat, after_attn_q_feat

