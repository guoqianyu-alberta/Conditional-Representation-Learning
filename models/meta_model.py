# meta-training model
from .components.DoubleCrossAttention import DoubleCrossAttention
from .components.SelfAttention import SelfAttention
from .components.DoubleSelfAttention import DoubleSelfAttention
import torch
from torch import nn
from .components.MLP import MLP
from .model_template import ModelTemplate


class MetaModel(ModelTemplate):
    def __init__(self, backbone, pretrained_path, input_channel, attn_hidden_dim, image_size, avg_pool, num_layer):
        super(MetaModel, self).__init__(backbone, pretrained_path, input_channel)
        conv_channel = int(image_size / 16)
        self.num_layer = num_layer

        self.double_cross_attn = nn.ModuleList(DoubleCrossAttention(input_channel=self.input_channel, hidden_dim=attn_hidden_dim, mode="sharing_weight") for i in range(self.num_layer))
        self.self_encoder = nn.ModuleList(SelfAttention(input_channel=self.input_channel, hidden_dim=attn_hidden_dim) for i in range(self.num_layer))
        self.mlp = nn.ModuleList(MLP(input_dim=self.input_channel+1, hidden_dim=2048, output_dim=self.input_channel, num_layers=3) for i in range(self.num_layer))
        # input_channesl = h = w
        self.conv = nn.ModuleList(nn.Sequential(
            nn.Conv3d(in_channels=conv_channel, out_channels=1, kernel_size=(conv_channel, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        ) for i in range(self.num_layer))

        self.classifier = nn.Linear(self.input_channel, 5)
        self.criterion = nn.CrossEntropyLoss()

        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = avg_pool
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, support_imgs, query_imgs, mode=None, label=None, is_feature=None):
        """
        :param support_imgs: shape = [bsz, n_way, 3, 224, 224]
        :param query_imgs: shpae = [bsz, n_way, 3, 224, 224]
        """
        # backbone s_feat, q_feat
        s_feat, q_feat = self.extract_features(support_imgs, query_imgs, mode, is_feature)
        s_feat_backbone = s_feat.clone()
        q_feat_backbone = q_feat[0].unsqueeze(0).clone()

        s_bsz, _, h, w = s_feat.shape
        q_bsz = q_feat.shape[0]

        for i in range(self.num_layer):
            # cross-attentions
            after_attn_support_feat, _ = self.double_cross_attn[i](s_feat.clone(), q_feat.clone())

            # get 4d correlation maps
            correlation_map_s = torch.einsum('bcxy, bchw -> bxyhw', q_feat.clone(), after_attn_support_feat)

            # [bsz, hq, wq, hs, ws] -> [bsz, 1, 1, hs, ws] -> [bsz, 1, h, w]
            corr_s = self.conv[i](correlation_map_s).view(correlation_map_s.shape[0], -1, h, w)

            # residual
            s_embedding = torch.concat((s_feat, corr_s), dim=1)

            # mlp
            s_embedding = self.mlp[i](s_embedding.flatten(2).permute(0, 2, 1))
            s_embedding = s_embedding.permute(0, 2, 1).view(-1, self.input_channel, h, w)  # [bsz, input_channel, h, w]

            #  self-attention
            s_embedding = self.self_encoder[i](s_embedding)
            s_embedding = s_embedding.view(s_bsz, -1, h, w)

            # next layer input
            s_feat = s_embedding

        # average pooling
        if self.avg_pool:
            s_embedding = self.avgpool(s_embedding)
            s_feat_backbone = self.avgpool(s_feat_backbone)
            q_feat_backbone = self.avgpool(q_feat_backbone)

        # add s_feat extracted from backbone
        # add q_feat extracted from backbone
        s_embedding = s_embedding.flatten(1)  # [n_way*n_shot, hidden_dim]
        s_feat_backbone = s_feat_backbone.flatten(1)
        q_feat_backbone = q_feat_backbone.flatten(1)

        # s_embedding = (s_embedding + s_feat_backbone) / 2

        # return self.get_loss_or_score(s_embedding, q_feat_backbone, label, mode)
        logits_s = self.classifier(torch.concat((s_feat_backbone, s_embedding)))  # directly obtain logits from model
        logits_q = self.classifier(q_feat_backbone)

        return logits_s, logits_q





