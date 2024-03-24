import math
import open_clip
import torch
from torch import nn
from torch.nn import CosineSimilarity
from torch.nn import MSELoss
from .backbone.Resnet12_IE import resnet12
from pynvml import *
from abc import abstractmethod
from FHC_Model.config import parse_args
import logging
import timm
from torchvision.models.resnet import resnet50


args = parse_args()


class ModelTemplate(nn.Module):
    def __init__(self, backbone, pretrained_path, input_channel, pretrained=True):
        super(ModelTemplate, self).__init__()
        if "CLIP_ViT" in backbone:
            clip, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
            self.feature_extractor = clip.visual
            self.input_channel = input_channel
        elif "vit_small" in backbone:
            self.feature_extractor = timm.create_model(
            'vit_small_patch16_224.dino',
                        pretrained=False,
                        checkpoint_path='/data/wujingrong/vit_small_patch16_224/pytorch_model.bin',
                        num_classes=0,  # remove classifier nn.Linear
        )
            self.input_channel = input_channel  # [6, 8, 8], [24, 4, 4], [96, 2, 2], [384, 1, 1]
        elif "Resnet12_IE" in backbone or "Resnet12" in backbone:
            self.feature_extractor = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, num_classes=64, no_trans=16, embd_size=64)
            self.input_channel = 640
            self.pretrained_path = pretrained_path
        elif "Resnet50" in backbone:    # [bsz, 2048, 7, 7]
            self.feature_extractor = torch.nn.Sequential(*(list(resnet50(pretrained=pretrained).children())[:-2]))
            self.input_channel = 2048
            self.pretrained_path = pretrained_path

        if "Resnet12" in backbone:   # Resnet50 and vit_small weights directly download from internet
            if self.pretrained_path:
                if self.pretrained_path.endswith('pth'):
                    ckpt = torch.load(self.pretrained_path)["model"]
                elif self.pretrained_path.endswith('.tar'):
                    ckpt = torch.load(self.pretrained_path)["state_dict"]
                new_state_dict = self.feature_extractor.state_dict()
                for k, v in ckpt.items():
                    name = k.replace("module.", "")
                    if name in list(new_state_dict.keys()):
                        new_state_dict[name] = v
                self.feature_extractor.load_state_dict(new_state_dict)
            elif pretrained:
                logging.info("pretrained")
            else:
                logging.info('No pretrained backbone')

        self.backbone = backbone

    def get_gpu_info(self):
        nvmlInit()
        gpu_id = torch.cuda.current_device()
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        print("gpu information")
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')

    def extract_features(self, support_img, query_img, mode=None, is_feature=None):
        assert mode in ["train", "test", "eval"]
        # extract feature from support image and query image
        if args.backbone_need_train:
            s_feat = self.feature_extractor(support_img)
            q_feat = self.feature_extractor(query_img)
        else:
            if not is_feature:
                with torch.no_grad():
                    s_feat = self.feature_extractor(support_img)
                    q_feat = self.feature_extractor(query_img)
            else:
                s_feat = support_img
                q_feat = query_img
        if self.backbone == "Resnet50":
            return s_feat, q_feat

        hidden_dim = s_feat.shape[-1]
        sequence_length = (int) (hidden_dim / self.input_channel)
        h = w = (int)(math.sqrt(sequence_length))

        s_bsz = s_feat.shape[0]
        q_bsz = q_feat.shape[0]

        s_feat = s_feat.view(s_bsz, self.input_channel, h, w)
        q_feat = q_feat.view(q_bsz, self.input_channel, h, w)
        return s_feat, q_feat

    def get_loss_or_score(self, s_embedding, q_embedding, label, mode):
        if mode == "train":
            dist = CosineSimilarity(dim=1, eps=1e-6)(q_embedding, s_embedding)
            mse = MSELoss()
            return mse(dist.float(), label.float())
        else:
            if args.n_shot > 1:
                return s_embedding, q_embedding
            score = CosineSimilarity(dim=1, eps=1e-6)(q_embedding, s_embedding)
            return score

    @abstractmethod
    def forward(self, support_img, query_img, label, mode=None, is_feature=None):
        pass