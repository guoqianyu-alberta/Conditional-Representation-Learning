# only backbone
import sys
sys.path.append('..')
from .model_template import ModelTemplate


class OnlyBackboneModel(ModelTemplate):
    def __init__(self, backbone, pretrained_path, input_channel):
        super(OnlyBackboneModel, self).__init__(backbone, pretrained_path, input_channel)

    def forward(self, support_img, query_img, label=None, mode=None, is_feature=None):
        s_feat, q_feat = self.extract_features(support_img, query_img, mode, is_feature)
        s_embedding = s_feat.flatten(1)
        q_embedding = q_feat.flatten(1)
        return self.get_loss_or_score(s_embedding, q_embedding, label, mode)


