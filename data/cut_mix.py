from random import randint
from random import uniform

import torch.nn


def mixup(img1, img2):
    lam = uniform(0.5, 1)
    output = img1.clone()
    output = output * lam + img2 * (1 - lam)
    return output


def cutmix(img1, img2):
    img_size = img1.shape[-1]
    min_length = img_size / 32
    max_length = img_size

    cutbox_h = randint(min_length, max_length)
    cutbox_w = randint(min_length, max_length)
    start_h = randint(0, img_size - cutbox_h)
    start_w = randint(0, img_size - cutbox_w)

    output = img1.clone()
    output[:, start_h:start_h + cutbox_h, start_w:start_w + cutbox_w] = img2[:, start_h:start_h+cutbox_h, start_w:start_w+cutbox_w]
    return output


class CutOut(torch.nn.Module):
    def forward(self, img):
        img_size = img.shape[-1]
        min_length = img_size / 32
        max_length = img_size

        cutbox_h = randint(min_length, max_length)
        cutbox_w = randint(min_length, max_length)
        start_h = randint(0, img_size - cutbox_h)
        start_w = randint(0, img_size - cutbox_w)

        output = img
        output[:, start_h:start_h + cutbox_h, start_w:start_w + cutbox_w] = 0
        return output

