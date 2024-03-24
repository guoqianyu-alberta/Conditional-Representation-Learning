from .components.DoubleCrossAttention import DoubleCrossAttention
from .components.DoubleSelfAttention import DoubleSelfAttention
import torch
from torch import nn
from .components.MLP import MLP
from .model_template import ModelTemplate
import math


class NotShareWeightModel(ModelTemplate):
    def __init__(self, backbone, pretrained_path, input_channel, attn_hidden_dim, image_size, avg_pool, num_layer):
        super(NotShareWeightModel, self).__init__(backbone, pretrained_path, input_channel)
        if "Resnet12" in backbone:
            conv_channel = int(image_size / 16)  # feat size
        elif "Resnet50" in backbone:
            conv_channel = int(image_size / 32)
        elif "vit_small" in backbone:
            conv_channel = (int)(math.sqrt((int) (384 / input_channel)))
        self.num_layer = num_layer

        attn_hidden_dim = self.input_channel
        self.double_cross_attn = nn.ModuleList(DoubleCrossAttention(input_channel=self.input_channel, hidden_dim=attn_hidden_dim, mode="double") for i in range(self.num_layer))
        self.self_encoder = nn.ModuleList(DoubleSelfAttention(input_channel=self.input_channel, hidden_dim=attn_hidden_dim, mode="double") for i in range(self.num_layer))
        self.mlp1 = nn.ModuleList(MLP(input_dim=self.input_channel+1, hidden_dim=2048, output_dim=self.input_channel, num_layers=3) for i in range(self.num_layer))
        self.mlp2 = nn.ModuleList(
            MLP(input_dim=self.input_channel + 1, hidden_dim=2048, output_dim=self.input_channel, num_layers=3) for i in
            range(self.num_layer))
        # input_channesl = h = w
        self.conv1 = nn.ModuleList(nn.Sequential(
            nn.Conv3d(in_channels=conv_channel, out_channels=1, kernel_size=(conv_channel, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm2d(1),
            nn.BatchNorm3d(1),  # change to batchnorm 3d
            nn.ReLU()
        ) for i in range(self.num_layer))
        self.conv2 = nn.ModuleList(nn.Sequential(
            nn.Conv3d(in_channels=conv_channel, out_channels=1, kernel_size=(conv_channel, 3, 3), padding=(0, 1, 1)),
            # nn.BatchNorm2d(1),
            nn.BatchNorm3d(1),  # change to batchnorm 3d
            nn.ReLU()
        ) for i in range(self.num_layer))

        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = avg_pool

    def forward(self, support_img, query_img, label, mode=None, is_feature=None):
        s_feat, q_feat = self.extract_features(support_img, query_img, mode, is_feature)

        s_feat_backbone = s_feat.clone()
        q_feat_backbone = q_feat.clone()

        s_bsz, _, h, w = s_feat.shape
        q_bsz = q_feat.shape[0]

        for i in range(self.num_layer):
            # cross-attentions
            after_attn_support_feat, after_attn_query_feat = self.double_cross_attn[i](s_feat.clone(), q_feat.clone())

            # get 4d correlation maps
            correlation_map_q = torch.einsum('bchw, bcxy -> bhwxy', after_attn_support_feat, after_attn_query_feat)
            correlation_map_s = torch.einsum('bcxy, bchw -> bxyhw', after_attn_query_feat, after_attn_support_feat)

            #[bsz, hq, wq, hs, ws] -> [bsz, 1, 1, hs, ws] -> [bsz, 1, h, w]
            corr_s = self.conv1[i](correlation_map_s).view(correlation_map_s.shape[0], -1, h, w)
            corr_q = self.conv2[i](correlation_map_q).view(correlation_map_q.shape[0], -1, h, w)

            # residual
            s_embedding = torch.concat((s_feat, corr_s), dim=1)
            q_embedding = torch.concat((q_feat, corr_q), dim=1)

            # mlp
            s_embedding = self.mlp1[i](s_embedding.flatten(2).permute(0, 2, 1))
            s_embedding = s_embedding.permute(0, 2, 1).view(-1, self.input_channel, h, w)  # [bsz, input_channel, h, w]
            q_embedding = self.mlp2[i](q_embedding.flatten(2).permute(0, 2, 1))
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
            s_embedding = self.avgpool(s_embedding)
            q_embedding = self.avgpool(q_embedding)
            s_feat_backbone = self.avgpool(s_feat_backbone)
            q_feat_backbone = self.avgpool(q_feat_backbone)

        # add s_feat extracted from backbone
        # add q_feat extracted from backbone
        s_embedding = s_embedding.flatten(1)   # [n_way*n_shot, hidden_dim]
        q_embedding = q_embedding.flatten(1)   # [n_way*n_shot, hidden_dim]
        s_feat_backbone = s_feat_backbone.flatten(1)
        q_feat_backbone = q_feat_backbone.flatten(1)

        s_embedding = (s_embedding + s_feat_backbone) / 2
        q_embedding = (q_embedding + q_feat_backbone) / 2

        # 直接输出s_embedding, s_feat_backbone

        return self.get_loss_or_score(s_embedding, q_embedding, label, mode)

