from networks.CrossAttention import CrossAttention
from networks.SelfAttention import SelfAttention
import torch
from torch import nn
from .model_template import ModelTemplate
import math
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CRLNet(ModelTemplate):
    def __init__(self, backbone, pretrained_path, image_size, avg_pool=True, num_layer=1):
        super(CRLNet, self).__init__(backbone, pretrained_path)
        # conv_channel depicts feature size
        if backbone == "Resnet12":
            conv_channel = int(image_size / 16)  # feat size
        if backbone == "Resnet50":
            conv_channel = int(image_size / 32)
        if backbone == "ViT":
            conv_channel = (int)(math.sqrt((int) (384 / self.input_channel)))
        
        self.num_layer = num_layer # always set to 1

        attn_hidden_dim = self.input_channel
        self.double_cross_attn = nn.ModuleList(CrossAttention(input_channel=self.input_channel, hidden_dim=attn_hidden_dim) for i in range(self.num_layer))
        self.self_encoder = nn.ModuleList(SelfAttention(input_channel=self.input_channel, hidden_dim=attn_hidden_dim) for i in range(self.num_layer))
        self.mlp = nn.ModuleList(MLP(input_dim=self.input_channel+1, hidden_dim=2048, output_dim=self.input_channel, num_layers=3) for i in range(self.num_layer))
        # input_channesl = h = w
        self.conv = nn.ModuleList(nn.Sequential(
            nn.Conv3d(in_channels=conv_channel, out_channels=1, kernel_size=(conv_channel, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(1),  
            nn.ReLU()
        ) for i in range(self.num_layer))

        if avg_pool: # alway set to True
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.avg_pool = avg_pool


    def forward(self, support_img, query_img, label=None, split=None):
        s_feat, q_feat = self.extract_features(support_img, query_img, split)

        s_feat_backbone = s_feat.clone()
        q_feat_backbone = q_feat.clone()

        s_bsz, _, h, w = s_feat.shape
        q_bsz = q_feat.shape[0]

        if self.avg_pool:
            s_feat_backbone = self.avgpool(s_feat_backbone).flatten(1)
            q_feat_backbone = self.avgpool(q_feat_backbone).flatten(1)

        for i in range(self.num_layer):
            # get cross-attention feats
            after_attn_support_feat, after_attn_query_feat = self.double_cross_attn[i](s_feat.clone(), q_feat.clone())  # [b, c, h, w]

            # get 4d correlation maps
            correlation_map_q = torch.einsum('bchw, bcxy -> bhwxy', after_attn_support_feat, after_attn_query_feat)
            correlation_map_s = torch.einsum('bcxy, bchw -> bxyhw', after_attn_query_feat, after_attn_support_feat)

            #[bsz, hq, wq, hs, ws] -> [bsz, 1, 1, hs, ws] -> [bsz, 1, h, w]
            corr_s = self.conv[i](correlation_map_s).view(correlation_map_s.shape[0], -1, h, w)
            corr_q = self.conv[i](correlation_map_q).view(correlation_map_q.shape[0], -1, h, w)


            # residual
            s_embedding = torch.concat((s_feat, corr_s), dim=1)
            q_embedding = torch.concat((q_feat, corr_q), dim=1)

            # mlp
            s_embedding = self.mlp[i](s_embedding.flatten(2).permute(0, 2, 1))
            s_embedding = s_embedding.permute(0, 2, 1).view(-1, self.input_channel, h, w)  # [bsz, input_channel, h, w]
            q_embedding = self.mlp[i](q_embedding.flatten(2).permute(0, 2, 1))
            q_embedding = q_embedding.permute(0, 2, 1).view(-1, self.input_channel, h, w)  # [bsz, input_channel, h, w]

            #  self-attention
            s_embedding, q_embedding = self.self_encoder[i](s_embedding, q_embedding)
            s_embedding = s_embedding.view(s_bsz, -1 , h, w)
            q_embedding = q_embedding.view(q_bsz, -1, h, w)

            # next layer input
            s_feat = s_embedding
            q_feat = q_embedding

        # average pooling
        if self.avg_pool:
            s_embedding = self.avgpool(s_embedding).flatten(1)
            q_embedding = self.avgpool(q_embedding).flatten(1)

        s_embedding = (s_embedding + s_feat_backbone) / 2
        q_embedding = (q_embedding + q_feat_backbone) / 2

        return self.get_loss_or_score(s_embedding, q_embedding, label, split)

