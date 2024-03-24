import logging
import os
import sys
sys.path.append("..")
import neptune
from sklearn.linear_model import LogisticRegression
import numpy as np
import utils
from config import parse_args
from torch.utils.data.dataloader import DataLoader
from data.TestDataset import TestDataset, EpisodicBatchSampler
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
import torch
from models.only_backbone import OnlyBackboneModel
from models.test_model import TestModel
from models.complete_model import CompleteModel
from utils import get_logger
import datetime
from k_shot import kshot_test
from models.not_share_weight_model import NotShareWeightModel
import time

from models.backbone.Resnet12_IE import resnet12
from torchvision.models.resnet import resnet50
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
local_rank = int(os.environ["LOCAL_RANK"])
API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYWYzMTAyMy1jMzkyLTQzZGYtOThiMC0xZWIxMmZhODU3OTIifQ=="


def correct(args, scores, process_score=True):
    y_query = np.repeat(range(args.eval_n_way), args.n_query)
    if process_score:
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
    else:
        top1_correct = np.sum(scores == y_query)
    return float(top1_correct), len(y_query)


def evaluation(args, model, eval_loader, is_feature):
    utils.fix_randseed(10)
    acc_all = []
    iter_num = len(eval_loader)
    if args.model_architecture == "only_backbone" and args.n_shot > 1:
        if "Resnet12" in args.backbone:
            model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64, no_trans=16, embd_size=64)
        elif "Resnet50" in args.backbone:
            model = torch.nn.Sequential(*(list(resnet50().children())[:-2]))
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
        model.cuda()
    model.eval()

    start_time = time.perf_counter()  # start time

    for epoch, (support_imgs, query_imgs) in enumerate(eval_loader):
        if args.model_architecture == "only_backbone" and args.n_shot > 1:
            if not is_feature:
                support_imgs = support_imgs.cuda()
                support_imgs = support_imgs.view(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size)
                query_imgs = query_imgs.cuda()
                query_imgs = query_imgs.view(args.eval_n_way * args.n_query, 3, args.image_size, args.image_size)
                with torch.no_grad():
                    support_feats = model(support_imgs)  # n_way * n_shot
                    query_feats = model(query_imgs)  # n_way * n_query
            else:
                hidden_dim = support_imgs.shape[-1]
                support_feats = support_imgs.view(-1, hidden_dim)
                query_feats = query_imgs.view(-1, hidden_dim)

            if "Resnet50" in args.backbone:
                support_feats = support_feats.view(-1, 2048*7*7)
                query_feats = query_feats.view(-1, 2048*7*7)
            support_feats = support_feats.detach().cpu().numpy()
            query_feats = query_feats.detach().cpu().numpy()

            support_labels = np.repeat(range(args.eval_n_way), args.n_shot)
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                                     multi_class='multinomial')
            clf.fit(support_feats, support_labels)
            scores = clf.predict(query_feats)

            # correct_this：共有多少预测正确的标签 count_this: total query images
            correct_this_episode, count_this_episode = correct(args, scores, False)
            acc_all.append(correct_this_episode / count_this_episode * 100)
        else:
            if is_feature:
                hidden_dim = support_imgs.shape[-1]
                support_imgs = support_imgs.view(-1, hidden_dim)  # [bsz, hidden_dim]
                query_imgs = query_imgs.view(-1, hidden_dim)  # [bsz, hidden_dim]

            else:
                support_imgs = support_imgs.cuda()
                query_imgs = query_imgs.cuda()
                support_imgs = support_imgs.view(-1, 3, args.image_size, args.image_size)  # [args.eval_shot * args.n_shot]
                query_imgs = query_imgs.view(-1, 3, args.image_size, args.image_size)  # [args.eval_shot * args.n_query]

            scores = []

            if args.n_shot == 1:
                for j in range(args.n_query * args.eval_n_way):
                    # score = [model(support_img=support_imgs[i].unsqueeze(0), query_img=query_imgs[j].unsqueeze(0), label=None, mode="eval", is_feature=is_feature) for i in
                    #          range(args.eval_n_way * args.n_shot)]
                    if not is_feature:
                        score = model(support_img=support_imgs, query_img=query_imgs[j].expand(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size), label=None, mode="eval", is_feature=is_feature)
                    else:
                        cur_query_img = query_imgs[j].expand(args.eval_n_way * args.n_shot, query_imgs.shape[-1])
                        score = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval", is_feature=is_feature)
                    scores.append(score)

                # scores = torch.stack(scores, dim=0).squeeze(3).squeeze(2)
                scores = torch.stack(scores, dim=0)
                correct_this_episode, count_this_episode = correct(args, scores)  # correct_this：共有多少预测对的标签 count_this: total query images
                acc_all.append(correct_this_episode / count_this_episode * 100)
            else: # kshot
                scores = kshot_test(model, support_imgs, query_imgs, is_feature=is_feature, method=args.method)
                correct_this_episode, count_this_episode = correct(args, scores, process_score=False)
                acc_all.append(correct_this_episode / count_this_episode * 100)

        total_time = time.perf_counter() - start_time

        if (epoch+1) % 1 == 0:
            logging.info('%d Test Acc = %4.2f%%' % (epoch+1, np.mean(np.asarray(acc_all))))
            if not args.logging:
                print('%d Test Acc = %4.2f%%' % (epoch+1, np.mean(np.asarray(acc_all))))


    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    logging.info('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    if not args.logging:
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

    if args.ablation and args.ablation_type == "kshot_time":
        return acc_mean, 1.96 * acc_std / np.sqrt(iter_num), total_time
    else:
        return acc_mean, 1.96 * acc_std / np.sqrt(iter_num)


if __name__ == '__main__':
    global args
    args = parse_args()

    get_logger(args, mode='test')

    assert args.model_architecture in ["only_backbone", "complete", "test_model", "not_share_weight"]
    if args.model_architecture == "only_backbone":
        model = OnlyBackboneModel(args.backbone, args.pretrained_path, args.input_channel)
    elif args.model_architecture == "complete":
        model = CompleteModel(args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim, args.image_size, args.avg_pool, args.num_layer)
    elif args.model_architecture == "test_model":
        model = TestModel(args, args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim, args.image_size, args.avg_pool, args.num_layer)
    elif args.model_architecture == "not_share_weight":
        model = NotShareWeightModel(args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim,
                              args.image_size, args.avg_pool, args.num_layer)
    else:
        raise ValueError('No such model')



    # only backbone just load pretrained backbone in model
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1000000))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    if not args.model_architecture == "only_backbone":
        model.load_state_dict(torch.load(args.load))
        """
        ckpt = torch.load(args.load)
        new_state_dict = model.state_dict()
        for k, v in ckpt.items():
            name = k.replace("module.", "")
            if name in list(new_state_dict.keys()):
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)\
        """


    if args.neptune:
        run = neptune.init_run(project="wujr/Model", api_token=API_TOKEN, source_files=["**/*.py", "**/*.sh"])
        run["parameters"] = args
        run["sys/tags"].add("test")
        run["sys/tags"].add(args.model_architecture)
        logging.info(run["sys/id"].fetch())

    transform = Compose([Resize((args.image_size)), CenterCrop((args.image_size)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    for e_dataset in args.eval_dataset:
        eval_dataset = TestDataset(e_dataset, args.eval_dataset_path, transform, args.n_query, args.n_shot,
                                   hdf5_file_path=args.eval_hdf5_file_path, backbone=args.backbone, is_feature=args.eval_is_feature)
        eval_sampler = EpisodicBatchSampler(len(eval_dataset), args.eval_n_way, args.eval_episodes)
        eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, num_workers=0, pin_memory=False)

        for epoch in range(1):
            # with torch.no_grad():
            if args.ablation and args.ablation_type == "kshot_time":
                val_acc, val_acc_std, test_time = evaluation(args, model, eval_loader, is_feature=args.eval_is_feature)
            else:
                val_acc, val_acc_std = evaluation(args, model, eval_loader, is_feature=args.eval_is_feature)
            if args.neptune:
                run["result"] = val_acc

        if not args.ablation:
            with open('test_results.txt' , 'a') as f:
                timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                exp_setting = '%s-%s-%s %sshot_test' %(args.backbone, args.model_architecture, e_dataset, args.n_shot)
                acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(args.eval_episodes, val_acc, val_acc_std)
                f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
        else:
            abfile = args.ablation_type + "_ablation_results"
            with open(abfile, 'a') as f:
                timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                if args.ablation_type == "augmentation":
                    exp_setting = '%s-%s %sshot_test: augmentation %s' % (args.backbone, e_dataset, args.n_shot, args.train_transform_method)
                elif args.ablation_type == "weight_share":
                    exp_setting = 'Non weight sharing model: %s-%s %sshot_test' % (args.backbone, e_dataset, args.n_shot)
                elif args.ablation_type == "residual":
                    exp_setting = 'Non residual model: %s-%s %sshot_test' % (
                    args.backbone, e_dataset, args.n_shot)
                elif args.ablation_type == "kshot_time":
                    exp_setting = 'Method %s: %s-%s %sshot test_time %s' % (
                        args.method, args.backbone, e_dataset, args.n_shot, test_time)
                elif args.ablation_type == "kshot":
                    exp_setting = 'Method %s: %s-%s %sshot' % (
                        args.method, args.backbone, e_dataset, args.n_shot)
                elif args.ablation_type == "module":
                    if args.module_type == 1:
                        modules = "Backbone + Cross-Attention: "
                    elif args.module_type == 2:
                        modules = "Backbone + Cross-Attention + 4D"
                    exp_setting = '%s %s-%s %sshot_test' % (modules,
                        args.backbone, e_dataset, args.n_shot)
                acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (args.eval_episodes, val_acc, val_acc_std)
                f.write('Time: %s, Setting: %s, Acc: %s \n' % (timestamp, exp_setting, acc_str))

