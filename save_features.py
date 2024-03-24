import os
import glob
import h5py
import torch.cuda
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from config import parse_args
from torch.autograd import Variable
import open_clip
args = parse_args()
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
from models.backbone.Resnet12_IE import resnet12


transform = Compose([Resize((args.image_size)), CenterCrop((args.image_size)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def img_to_feat(imgs, model):
    imgs = imgs.cuda()
    img_var = Variable(imgs)
    feats = model(img_var)
    return feats


def save_train_features_classwise(model, metadata_classwise, outfile_metadata_classwise):

    f_class_wise = h5py.File(outfile_metadata_classwise, 'w')
    class_list = list(metadata_classwise.keys())
    batch_count = 1
    for cl in class_list:
        print('{:d}/{:d}'.format(batch_count, len(class_list)))
        batch_count = batch_count + 1
        cls_data = metadata_classwise[cl]

        cls_dataset = ClassDataset(cls_data, transform=transform)
        cls_dataloader = DataLoader(cls_dataset, batch_size=20, shuffle=False, drop_last=False)

        # save classwise meta feats
        count = 0
        cls_feats = None
        for i, imgs in tqdm.tqdm(enumerate(cls_dataloader)):
            feats = img_to_feat(imgs, model)
            if cls_feats is None:
                cls_feats = f_class_wise.create_dataset(cl, [len(cls_data)] + list(feats.size()[1:]), dtype='f')
            cls_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
            count = count + feats.size(0)

    f_class_wise.close()
    """

    f_complete = h5py.File(outfile_metadata, 'w')
    complete_feats = None
    complete_data = metadata
    complete_dataset = ClassDataset(complete_data, transform=transform)
    complete_loader = DataLoader(complete_dataset, batch_size=30, shuffle=False, drop_last=False)

    count = 0
    for i, imgs in tqdm.tqdm(enumerate(complete_loader)):
        feats = img_to_feat(imgs, model)

        if complete_feats is None:
            complete_feats = f_complete.create_dataset("complete", [len(complete_data)] + list(feats.size()[1:]), dtype='f')
        complete_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        count = count + feats.size(0)

    f_complete.close()
    """

def save_test_features(model, metadata_classwise, outfile):
    f = h5py.File(outfile, 'w')
    class_list = list(metadata_classwise.keys())
    i = 1
    for cl in class_list:
        print('{:d}/{:d}'.format(i, len(class_list)))
        i = i + 1
        sup_data = metadata_classwise[cl][0]
        que_data = metadata_classwise[cl][1]

        sup_dataset = ClassDataset(sup_data, transform=transform)
        sup_dataloader = DataLoader(sup_dataset, batch_size=25, shuffle=False, drop_last=False)
        que_dataset = ClassDataset(que_data, transform=transform)
        que_dataloader = DataLoader(que_dataset, batch_size=25, shuffle=False, drop_last=False)

        # save support feats
        count = 0
        cls_sup_feats = None
        for i, imgs in tqdm.tqdm(enumerate(sup_dataloader)):
            feats = img_to_feat(imgs, model)
            if cls_sup_feats is None:
                cls_sup_feats = f.create_dataset(cl + "_support", [len(sup_data)] + list(feats.size()[1:]), dtype='f')
            cls_sup_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            count = count + feats.size(0)

        # save query feats
        count = 0
        cls_que_feats = None
        for i, imgs in tqdm.tqdm(enumerate(que_dataloader)):
            feats =  img_to_feat(imgs, model)
            if cls_que_feats is None:
                cls_que_feats = f.create_dataset(cl + "_query", [len(que_data)] + list(feats.size()[1:]), dtype='f')
            cls_que_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            count = count + feats.size(0)

    f.close()

def get_features(hdf5_file,class_list):
    with h5py.File(hdf5_file, 'r') as f:
        cl_feat_data = {}
        for cl in class_list:
            cl_feat_data[cl] = []

            sup_feat = f[cl+"_support"][...]
            que_feat = f[cl+"_query"][...]

            cl_feat_data[cl].append(sup_feat)
            cl_feat_data[cl].append(que_feat)

    return cl_feat_data


class ClassDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def getImage(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, i):
        img_path = self.dataset[i]
        img = self.getImage(img_path)
        return img

    def __len__(self):
        return len(self.dataset)


def save_train_features(model, metadata, outfile_metadata):
    f_complete = h5py.File(outfile_metadata, 'w')
    complete_feats = None
    complete_data = metadata
    complete_dataset = ClassDataset(complete_data, transform=transform)
    complete_loader = DataLoader(complete_dataset, batch_size=1, shuffle=False, drop_last=False)

    count = 0
    for i, imgs in tqdm.tqdm(enumerate(complete_loader)):
        feats = img_to_feat(imgs, model)

        if complete_feats is None:
            complete_feats = f_complete.create_dataset("complete", [len(complete_data)] + list(feats.size()[1:]), dtype='f')
        complete_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        count = count + feats.size(0)

    f_complete.close()

if __name__ == '__main__':
    """
    clip, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained_path)
    model = clip.visual
    """
    model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64, no_trans=16,
                                      embd_size=64)
    if args.pretrained_path.endswith('pth'):
        ckpt = torch.load(args.pretrained_path)["model"]
    elif args.pretrained_path.endswith('.tar'):
        ckpt = torch.load(args.pretrained_path)["state_dict"]
    new_state_dict = model.state_dict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        if name in list(new_state_dict.keys()):
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    device = torch.device("cuda")
    model.to(device)
    model.eval()


    with open('./data/data_files/plant_virus.pkl', 'rb') as f:
        metadata_classwise = pickle.load(f)

    outfile_metadata_classwise = os.path.join( './features', 'Resnet12_IE', 'test', 'plant_virus' + ".hdf5")

    dirname = os.path.dirname(outfile_metadata_classwise)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    save_test_features(model, metadata_classwise, outfile_metadata_classwise)
    # save_train_features_classwise(model, metadata_classwise, outfile_metadata_classwise)

    """

    with open('/data/wujingrong/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    outfile_metadata = os.path.join( './features', 'train', 'ImageNet' + ".hdf5")

    dirname = os.path.dirname(outfile_metadata)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    save_train_features(model, metadata, outfile_metadata)
    """
