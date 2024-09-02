import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= 'CRLNet model configuration')
    parser.add_argument('--gpu', type=int, default=0)

    # model
    parser.add_argument('--backbone', default='Resnet12', type=str, choices=['Resnet12', 'Resnet50', 'ViT'])
    parser.add_argument('--pretrained_path', default=None, type=str)

    # train
    parser.add_argument('--lr', default = 0.0001, type=float)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--train_dataset', default=None, type=str, choices=['miniImageNet', 'ImageNet'])
    parser.add_argument('--train_dataset_path', default="/data/wujingrong", type=str)
    parser.add_argument('--train_bsz', default=12, type=int)
    parser.add_argument('--epochs', default=500, type=int)

    # test
    parser.add_argument('--eval_dataset', type=str, choices=['animal', 'insect', 'minet', 'plant_virus', 'oracle', 'fungus'])
    parser.add_argument('--eval_dataset_path', default="/data/wujingrong/Classification_Datasets/data_files", type=str)
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--eval_episodes', default=600, type=int)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--log_path', default='log', type=str)

    return parser.parse_args()