# position encoding for transformers, from mask2former github
import torch
import math
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) # [bsz, H, W]
        not_mask = ~mask # [1, 16, 16] * True
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # [1, 16, 16]
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # [1, 16, 16]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # [hidden_dim, ]

        pos_x = x_embed[:, :, :, None] / dim_t # [bsz, H, W, hidden_dim]
        pos_y = y_embed[:, :, :, None] / dim_t # [bsz, H, W, hidden_dim]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) #[bsz, 2*hidden_dim, H, W]
        return pos

if __name__ == '__main__':
    pe_layer = PositionEmbeddingSine(normalize=True)
    x = torch.randn(1, 256, 16, 16)
    pos = pe_layer(x, None)