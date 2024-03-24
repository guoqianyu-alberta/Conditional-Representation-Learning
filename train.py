from test import evaluation
# from baseline_model import Baseline
from models.only_backbone import OnlyBackboneModel
from models.complete_model import CompleteModel
from models.test_model import TestModel
from models.not_share_weight_model import NotShareWeightModel
from config import parse_args
import torch.optim as optim
from data.TrainDataset import TrainDataset
from torch.utils.data.dataloader import DataLoader
from data.TestDataset import TestDataset, EpisodicBatchSampler
import datetime
from torch.utils.data.distributed import DistributedSampler as Sampler
import torch
import neptune
import os
import logging
from data.augmentation import augmentation
from utils import get_logger
from torchvision.transforms import v2
from torch.utils.data import default_collate

os.environ['CUDA_VISIBLE_DEVICES']= '1, 2, 3, 5'
local_rank = int(os.environ["LOCAL_RANK"])
API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYWYzMTAyMy1jMzkyLTQzZGYtOThiMC0xZWIxMmZhODU3OTIifQ=="


def main_process(local_rank):
    return local_rank == 0


if __name__ == "__main__":
    global args
    args = parse_args()
    get_logger(args, mode='train')

    assert args.model_architecture in ["only_backbone", "complete", "test_model", "not_share_weight"]
    if args.model_architecture == "only_backbone":
        model = OnlyBackboneModel(args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim)
    elif args.model_architecture == "complete":
        model = CompleteModel(args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim, args.image_size, args.avg_pool, args.num_layer)
    elif args.model_architecture == "test_model":
        model = TestModel(args, args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim,
                              args.image_size, args.avg_pool, args.num_layer)
    elif args.model_architecture == "not_share_weight":
        model = NotShareWeightModel(args.backbone, args.pretrained_path, args.input_channel, args.attn_hidden_dim,
                              args.image_size, args.avg_pool, args.num_layer)
    else:
        raise ValueError('No such model')

    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1000000))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

    if args.resume:
        logging.info("Resume Training")
        if args.load is None:
            raise ValueError("No Loading Model......")
        else:
            ckpt = torch.load(args.load)
            new_state_dict = model.state_dict()
            for k, v in ckpt.items():
                name = k.replace("module.", "")
                if name in list(new_state_dict.keys()):
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            logging.info("Loading Model from %s", args.load)

    optimizer = optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": 0.05}])

    if args.neptune and main_process(local_rank):
        run = neptune.init_run(project="wujr/Model", api_token=API_TOKEN, source_files=["**/*.py", "**/*.sh"])
        run["parameters"] = args
        run["sys/tags"].add("train")
        run["sys/tags"].add(args.model_architecture)
        run["sys/tags"].add(str(args.num_layer) + " layer")
        run['sys/tags'].add(args.transform_method + " transform method")
        if args.ablation:
            run["sys/tags"].add("ablation")
        logging.info(run["sys/id"].fetch())

    # transform = Compose([Resize((args.image_size)), CenterCrop((args.image_size)), ToTensor(),
    #                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform = augmentation(args.image_size, args.transform_method)
    transform_eval = transform["support"]
    train_dataset = TrainDataset(args.train_dataset,
                                 args.train_dataset_path,
                                 args.train_hdf5_file_path,
                                 transform,
                                 args.transform_method,
                                 is_feature=args.train_is_feature)

    train_sampler = Sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_bsz, sampler=train_sampler, num_workers=0, pin_memory=False)

    dataset_name = ['animal', 'insect', 'minet', 'oracle', 'fungus', 'plant_virus']
    eval_dataset = []
    eval_sampler = []
    eval_loader = []
    for i in range(len(dataset_name)):
        eval_dataset.append(TestDataset(dataset_name[i],
                                   args.eval_dataset_path,
                                   transform_eval,
                                   args.n_query,
                                   args.n_shot,
                                   hdf5_file_path=args.eval_hdf5_file_path,
                                   backbone=args.backbone,
                                   is_feature=args.eval_is_feature))
        eval_sampler.append(EpisodicBatchSampler(len(eval_dataset[i]), args.eval_n_way, args.eval_episodes))
        eval_loader.append(DataLoader(eval_dataset[i], batch_sampler=eval_sampler[i], num_workers=0, pin_memory=False))

    if "Resnet50" in  args.backbone:
        sota_acc = [38, 39, 38, 33, 31, 66]
    elif "Resnet12" in args.backbone:
        sota_acc = [31, 35, 38, 26, 29, 57]
    elif "vit_small" in args.backbone:
        sota_acc = [43, 48, 44, 28.6, 34, 71]
    best_val_acc = [float('-inf')] * len(dataset_name)
    best_sum = float('-inf')
    best_better_datasets = 0
    print_freq = 10
    save_freq = 10

    for epoch in range(args.epochs):
        total_loss = 0
        train_loader.sampler.set_epoch(epoch)
        for i, (support_images, query_images, label) in enumerate(train_loader):
            support_images = support_images.cuda()
            query_images = query_images.cuda()
            label = label.cuda()

            model.train()
            loss = model(support_images, query_images, label, mode='train', is_feature=args.train_is_feature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss
            if i % print_freq == 0 and i != 0 and main_process(local_rank):
                logging.info('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), total_loss/float(i+1)))
                if not args.logging:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), total_loss/float(i+1)))

            if args.train_dataset == "ImageNet" and i % 50 == 0 and i != 0 and main_process(local_rank):
                if args.neptune and main_process(local_rank):
                    run['loss'].append(total_loss / float(i + 1))
                model.eval()
                with torch.no_grad():
                    val_accs = []
                    sum = 0
                    better_datasets = 0
                    for i in range(len(eval_dataset)):
                        logging.info(dataset_name[i])
                        val_acc, val_acc_std = evaluation(args, model, eval_loader[i], is_feature=args.eval_is_feature)
                        val_accs.append(val_acc)
                        run[dataset_name[i] + "_acc"].append(val_acc)
                        if val_acc > sota_acc[i]:
                            better_datasets = better_datasets + 1
                        sum = sum + val_acc
                    if better_datasets > best_better_datasets or (
                            better_datasets == best_better_datasets and sum > best_sum):
                        for i in range(len(val_accs)):
                            run[dataset_name[i] + "best_acc"] = val_accs[i]
                        best_better_datasets = better_datasets
                        best_sum = max(sum, best_sum)
                        torch.save(model.state_dict(), os.path.join(args.log_path, 'best_model.pt'))
                        run['best_better_datasets'] = best_better_datasets

        if args.neptune and main_process(local_rank):
            run['loss'].append(total_loss/float(i+1))

        if main_process(local_rank):
            model.eval()
            with torch.no_grad():
                val_accs = []
                sum = 0
                better_datasets = 0
                for i in range(len(eval_dataset)):
                    logging.info(dataset_name[i])
                    val_acc, val_acc_std = evaluation(args, model, eval_loader[i], is_feature=args.eval_is_feature)
                    val_accs.append(val_acc)
                    run[dataset_name[i] + "_acc"].append(val_acc)
                    if val_acc > sota_acc[i]:
                        better_datasets = better_datasets + 1
                    sum = sum + val_acc
                if better_datasets > best_better_datasets or (
                        better_datasets == best_better_datasets and sum > best_sum):
                    for i in range(len(val_accs)):
                        run[dataset_name[i] + "best_acc"] = val_accs[i]
                    best_better_datasets = better_datasets
                    best_sum = max(sum, best_sum)
                    torch.save(model.state_dict(), os.path.join(args.log_path, 'best_model.pt'))
                    run['best_better_datasets'] = best_better_datasets
                # val_acc, val_acc_std = evaluation(args, model, eval_loader, is_feature=args.eval_is_feature)
            """
            if args.neptune:
                run['val_acc'].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                run['best_val_acc'] = best_val_acc
                run['best_epoch'] = epoch
                torch.save(model.state_dict(), os.path.join(args.log_path, 'best_model.pt'))
            """


