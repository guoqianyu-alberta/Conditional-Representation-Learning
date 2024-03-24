from urllib.request import urlopen
from PIL import Image
import timm
import torch


if __name__ == "__main__":
    model = timm.create_model(
        'vit_small_patch16_224.dino',
        pretrained=False,
        checkpoint_path='/data/wujingrong/vit_small_patch16_224/pytorch_model.bin',
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()
    output = model(torch.randn(1, 3,224, 224))  # output is (batch_size, num_features) shaped tensor
    print(output.shape)