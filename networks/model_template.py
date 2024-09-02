import math
import torch
from torch import nn
from torch.nn import CosineSimilarity
from torch.nn import MSELoss
from .backbone.Resnet12 import resnet12
from pynvml import *
from abc import abstractmethod
from config import parse_args
import timm
from torchvision.models.resnet import resnet50


args = parse_args()

class ModelTemplate(nn.Module):
    def __init__(self, backbone, pretrained_path):
        super(ModelTemplate, self).__init__()
        
        if backbone == "ViT": 
            self.feature_extractor = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0)
            self.input_channel = 384  # [6, 8, 8], [24, 4, 4], [96, 2, 2], [384, 1, 1]
        
        if backbone == "Resnet12":
            self.feature_extractor = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, num_classes=64, no_trans=16, embd_size=64)
            self.input_channel = 640
            ckpt = torch.load(pretrained_path)["state_dict"]
            new_state_dict = self.feature_extractor.state_dict()
            for k, v in ckpt.items():
                name = k.replace("module.", "")
                if name in list(new_state_dict.keys()):
                    new_state_dict[name] = v
            self.feature_extractor.load_state_dict(new_state_dict)
        
        if backbone == "Resnet50":    
            self.feature_extractor = torch.nn.Sequential(*(list(resnet50(pretrained=True).children())[:-2]))
            self.input_channel = 2048

    def extract_features(self, support_img, query_img, split):
        assert split in ["train", "test", "eval"]

        with torch.no_grad():
            s_feat = self.feature_extractor(support_img)
            q_feat = self.feature_extractor(query_img)

        if len(s_feat.shape) == 4:
            return s_feat, q_feat

        hidden_dim = s_feat.shape[-1]
        sequence_length = (int) (hidden_dim / self.input_channel)
        h = w = (int)(math.sqrt(sequence_length))

        s_bsz = s_feat.shape[0]
        q_bsz = q_feat.shape[0]

        s_feat = s_feat.view(s_bsz, self.input_channel, h, w)
        q_feat = q_feat.view(q_bsz, self.input_channel, h, w)
        return s_feat, q_feat

    def get_loss_or_score(self, s_embedding, q_embedding, label, split):
        if split == "train":
            dist = CosineSimilarity(dim=1, eps=1e-6)(q_embedding, s_embedding)
            mse = MSELoss()
            return mse(dist.float(), label.float())
        else:
            if args.n_shot > 1:
                return s_embedding, q_embedding
            score = CosineSimilarity(dim=1, eps=1e-6)(q_embedding, s_embedding)
            return score

    @abstractmethod
    def forward(self, support_img, query_img, label, split=None):
        pass