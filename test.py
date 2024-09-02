import logging
import sys
import tqdm

sys.path.append("..")
import numpy as np
import utils
from config import parse_args
from torch.utils.data.dataloader import DataLoader
from data.TestDataset import TestDataset, EpisodicBatchSampler
import torch
from networks.CRLNet import CRLNet
from utils import get_logger
from k_shot import kshot_test
import time

def correct(scores, process_score=True):
    y_query = np.repeat(range(args.n_way), args.n_query)
    if process_score:
        _, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
    else:
        top1_correct = np.sum(scores == y_query)
    return float(top1_correct), len(y_query)


def evaluation(model, eval_loader):
    utils.fix_randseed(10)
    acc_all = []
    iter_num = len(eval_loader)

    for epoch, (support_imgs, query_imgs) in tqdm.tqdm(enumerate(eval_loader)):

        support_imgs = support_imgs.cuda(device)
        query_imgs = query_imgs.cuda(device)
        support_imgs = support_imgs.view(-1, 3, args.image_size, args.image_size)  # [args.eval_shot * args.n_shot]
        query_imgs = query_imgs.view(-1, 3, args.image_size, args.image_size)  # [args.eval_shot * args.n_query]

        scores = []
        if args.n_shot == 1:
            for j in range(args.n_query * args.n_way): 
                score = model(support_img=support_imgs, query_img=query_imgs[j].expand(args.n_way * args.n_shot, 3,
                args.image_size, args.image_size), label=None, split="eval")
                scores.append(score)
            scores = torch.stack(scores, dim=0)
            correct_this_episode, count_this_episode = correct(scores)  
            acc_all.append(correct_this_episode / count_this_episode * 100)
        
        else: # kshot
            scores = kshot_test(model, support_imgs, query_imgs)
            correct_this_episode, count_this_episode = correct(scores, process_score=False)
            acc_all.append(correct_this_episode / count_this_episode * 100)

        if (epoch+1) % 1 == 0:
            logging.info('Test Acc = %4.2f%%' % (np.mean(np.asarray(acc_all))))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    logging.info('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    
    return acc_mean, 1.96 * acc_std / np.sqrt(iter_num)


if __name__ == '__main__':
    global args
    args = parse_args()

    print("Getting Logger...")
    get_logger(args, mode='test')

    # initiate model
    print("Initiating Model...")
    model = CRLNet(args.backbone, args.pretrained_path, args.image_size)

    # load model
    print("Loading Model...")
    state_dict = {}
    ckpt = torch.load(args.load)
    ckpt_keys = list(torch.load(args.load).keys())
    for key in model.state_dict().keys():
        tempKey = "module." + key
        if tempKey in ckpt_keys: state_dict[key] = ckpt[tempKey]
    model.load_state_dict(state_dict)

    # model to cuda
    global device
    device = torch.device(args.gpu)
    model.to(device)
    model.eval()

    # construct data
    print(f"Constructing Dataset {args.eval_dataset}")
    eval_dataset = TestDataset(args.eval_dataset, args.eval_dataset_path, args.n_query, args.n_shot)
    eval_sampler = EpisodicBatchSampler(len(eval_dataset), args.n_way, args.eval_episodes)
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, num_workers=0, pin_memory=False)

    print(f"Starting Evaluating {args.eval_dataset}")
    val_acc, val_acc_std = evaluation(model, eval_loader)

    print("Writing Result...")
    with open('results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = '%s-%s %sshot_test' %(args.backbone, args.eval_dataset, args.n_shot)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(args.eval_episodes, val_acc, val_acc_std)
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
    print("Done!")
