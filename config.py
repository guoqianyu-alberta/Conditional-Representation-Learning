import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= 'FHC model configuration')
    parser.add_argument('--neptune', action='store_true')
    # model
    parser.add_argument('--backbone', default='ViT-g-14', type=str)
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--input_channel', default=1408, type=int)
    parser.add_argument('--attn_hidden_dim', default=640, type=int)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--model_description', default=None, type=str)
    parser.add_argument('--avg_pool', action='store_true')

    # train
    parser.add_argument('--lr', default = 0.0001, type=float)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--train_dataset', default=None, type=str, choices=['miniImageNet', 'ImageNet', 'CompleteDataset'])
    parser.add_argument('--train_dataset_path', default="/data/wujingrong", type=str)
    parser.add_argument('--train_bsz', default=12, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train_hdf5_file_path', default='features/Resnet12/train', type=str)
    parser.add_argument('--train_is_feature', action='store_true')
    parser.add_argument('--model_architecture', default=None, type=str)
    parser.add_argument('--transform_method', default=None, type=str, choices=['A', 'B', 'C', 'D', 'E', 'F'])
    parser.add_argument('--backbone_need_train', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # meta_train
    parser.add_argument('--train_n_way', default=5, type=int)
    parser.add_argument('--train_episodes', type=int)
    parser.add_argument('--train_n_shot', default=1, type=int)
    parser.add_argument('--update_freq', type=int)
    parser.add_argument('--update_step', default=5, type=int)
    parser.add_argument('--task_num', default=4, type=int)
    parser.add_argument('--update_lr', default=0.01, type=float)

    # eval/test
    #  datasets = ['animal', 'insect', 'minet', 'oracle', 'fungus', 'plant_virus']
    datasets = ['animal', 'insect', 'minet']
    parser.add_argument('--eval_dataset', default=datasets, nargs='+')
    parser.add_argument('--eval_dataset_path', default="data/data_files", type=str)
    parser.add_argument('--eval_hdf5_file_path', default='features', type=str)
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument('--n_shot', default=10, type=int)
    parser.add_argument('--eval_n_way', default=5, type=int)
    parser.add_argument('--eval_episodes', default=600, type=int)
    parser.add_argument('--eval_is_feature', action='store_true')
    parser.add_argument('--method', default=None, type=int)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--log_path', default='log', type=str)
    parser.add_argument('--train_transform_method', type=str, choices=['A', 'B', 'D'])
    parser.add_argument('--logging', action='store_true')

    # ablation
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--ablation_type', type=str, choices=["module", "residual", "augmentation", "kshot", "weight_share", "kshot_time", "kshot"])
    parser.add_argument('--module_type', type=int)

    return parser.parse_args()