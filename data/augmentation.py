# Data Augmentation
from torchvision.transforms import v2
from data.cut_mix import CutOut


def augmentation(img_size, transform_method):
    assert transform_method in ['A', 'B', 'C', 'D', 'E', 'F']

    transform = {}
    transform["support"] = v2.Compose([
            v2.Resize(size=img_size),
            v2.CenterCrop(size=img_size),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if transform_method == 'A':
        transform["query"] = v2.Compose([
            v2.RandomResizedCrop(size=img_size, antialias=True),
            # Crop a random portion of image and resize it to a given size
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            v2.RandomHorizontalFlip(),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    elif transform_method == 'B':
        transform["query"] = v2.Compose([
            v2.RandomResizedCrop(size=img_size, antialias=True),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif transform_method == 'C':
        transform["query"] = v2.Compose([
            v2.RandomResizedCrop(size=img_size, antialias=True),
            v2.RandAugment(),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif transform_method == 'D':
        transform['query'] = v2.Compose([
            v2.RandomResizedCrop(size=img_size, antialias=True),
            v2.TrivialAugmentWide(),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # cutmix or mixup
    elif transform_method == 'E':
        transform['query'] = v2.Compose([
            v2.RandomResizedCrop(size=img_size, antialias=True),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif transform_method == 'F':
        transform["query"] = v2.Compose([
            v2.RandomResizedCrop(size=img_size, antialias=True),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.RandomChoice([CutOut(), v2.RandAugment()]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


"""
v2.RandomRotation(degrees=(0, 360)),
v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
random erasing
"""