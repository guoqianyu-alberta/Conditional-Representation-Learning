r""" Helper functions """
import random
import torch
import numpy as np
import datetime
import os
import logging


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def get_logger(args, mode=None):
    assert mode in ['train', 'test']
    logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    backbone = args.backbone
    args.log_path = os.path.join(args.log_path, 'train', backbone, str(args.train_dataset) + "_" + logtime) if mode=='train' \
        else os.path.join(args.log_path, 'test', args.eval_dataset, backbone + '_' + str(args.n_shot) + 'shot' + logtime)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    logging.basicConfig(filemode='w',
                        filename=os.path.join(args.log_path, 'log.txt'),
                        level=logging.INFO,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M:%S')

    # Console log config
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Log arguments
    logging.info('\n:==================== Start =====================')
    for arg_key in args.__dict__:
        logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
    logging.info(':================================================\n')

