from torch.utils.data import Dataset
import pickle
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
import numpy as np
import os


class TestDataset(Dataset):
    def __init__(self, dataset, dataset_path, n_query, n_support):
        """
        :param dataset: test_dataset
        :param dataset_path: path to data
        :param n_query: how many query images per class
        :param n_support: how many support images per class
        """
        dataset_pkl = os.path.join('datafiles', dataset + ".pkl")
        self.dataset = dataset

        with open(dataset_pkl, 'rb') as f:
            self.meta = pickle.load(f)
        
        self.cl_list = list(self.meta.keys())
        self.n_query = n_query
        self.n_support = n_support
        self.path = dataset_path
        self.transform = Compose([Resize((224)), CenterCrop((224)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.cl_list)

    def getimage(self,image_path):
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, i):
        chosen_cls = self.cl_list[i]

        support_files = np.random.choice(self.meta[chosen_cls][0], self.n_support, replace=False)
        query_files = np.random.choice(self.meta[chosen_cls][1], self.n_query, replace=False)
        
        support_imgs = torch.stack([self.getimage(image_path) for image_path in support_files])
        query_imgs = torch.stack([self.getimage(image_path) for image_path in query_files])     

        return support_imgs, query_imgs


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # sample n_way from total classes
            yield torch.randperm(self.n_classes)[:self.n_way]