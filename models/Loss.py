import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    args:
        distance (function): A function that returns the distance between two tensors - should be a valid metric over R; default= L2 distance
        margin (scalar): The margin value between positive and negative class ; default=1.0
        miner (function, optional): A function that calculates similarity labels [0,1] on the input if no labels are explicitly provided - should return (embs1, embs2, labels)
    """

    def __init__(self,
                 distance=lambda x, y: torch.pow(x - y, 2).sum(1),
                 margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, x, y, label=None):
        """
        Return the contrastive loss between two similar or dissimilar outputs

        Args:
            x (torch.Tensor) : The first input tensor (B, N)
            y (torch.Tensor) : The second input tensor (B,N)
            label (torch.Tensor, optional) : A tensor with elements either 0 or 1 indicating dissimilar or similar (B, 1)
        """
        assert x.shape == y.shape, str(x.shape) + "does not match input 2: " + str(y.shape)

        # 没有开根号，就是距离的平方
        square_distance = self.distance(x, y)
        distance = torch.sqrt(square_distance)

        # When the label is 1 (similar) - the loss is the distance between the embeddings
        # When the label is 0 (dissimilar) - the loss is the distance between the embeddings and a margin
        loss_contrastive = torch.mean(0.5 * ((label) * square_distance +
                                             (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)))
        return loss_contrastive
