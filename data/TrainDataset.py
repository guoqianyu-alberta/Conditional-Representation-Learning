from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import torch
import os
from torchvision.transforms import v2


class TrainDataset(Dataset):
    def __init__(self, dataset, dataset_path):
        # need to construct two pkl files
        # metadata contains all images
        # metadata_classwise contains a dict with categories as key and images as value
        
        super(TrainDataset, self).__init__()

        assert dataset in ["ImageNet", "miniImageNet"]
        metadata_path = os.path.join(dataset_path, dataset + ".pkl")
        metadata_classwise_path = os.path.join(dataset_path, dataset + "_classwise.pkl")
        
        # all images
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f) 
        
        # images by category
        with open(metadata_classwise_path, 'rb') as f:
            self.metadata_classwise = pickle.load(f) 
        
        self.classes = list(self.metadata_classwise.keys())

        self.transform = {}
        self.transform["support"] = v2.Compose([
            v2.Resize(size=224),
            v2.CenterCrop(size=224),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform["query"] = v2.Compose([
            v2.RandomResizedCrop(size=224, antialias=True),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def getimage(self, image_path, split):
        assert split in ['support', 'query']
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform[split](img)
        return img

    def __getitem__(self, index):

        query_img_path = self.metadata[index]
        query_class = query_img_path.split('/')[4]
        query_img = self.getimage(query_img_path, "query")

        if index % 2 == 0: # positive
            support_class = query_class
            support_img_path = np.random.choice(self.metadata_classwise[support_class], 1).item()
            while support_img_path == query_img_path:
                support_img_path = np.random.choice(self.metadata_classwise[support_class], 1).item()

        else: # negative
            support_class = np.random.choice(self.classes, 1).item()
            while support_class == query_class:
                support_class = np.random.choice(self.classes, 1).item()
            support_img_path = np.random.choice(self.metadata_classwise[support_class], 1).item()

        support_img = self.getimage(support_img_path, "support")

        if support_class == query_class:
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)

        return support_img, query_img, label




