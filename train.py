from networks.CRLNet import CRLNet
from config import parse_args
import torch.optim as optim
from data.TrainDataset import TrainDataset
from torch.utils.data.dataloader import DataLoader
import datetime
from torch.utils.data.distributed import DistributedSampler as Sampler
import torch
import os
import logging
from utils import get_logger

os.environ['CUDA_VISIBLE_DEVICES']= '0, 4, 5, 6'
local_rank = int(os.environ["LOCAL_RANK"])


def main_process(local_rank):
    return local_rank == 0

if __name__ == "__main__":
    global args
    args = parse_args()
    get_logger(args, mode='train')

    model = CRLNet(args.backbone, args.pretrained_path, args.image_size)

    # model to multiple gpus
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1000000))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

    # set optimizer
    optimizer = optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": 0.05}])

    # set dataloader
    train_dataset = TrainDataset(args.train_dataset, args.train_dataset_path)
    train_sampler = Sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_bsz, sampler=train_sampler, num_workers=0, pin_memory=False)

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
            loss = model(support_images, query_images, label, split='train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss

            if i % print_freq == 0 and main_process(local_rank):
                logging.info('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), total_loss/float(i+1)))

            if i % save_freq == 0 and main_process(local_rank):
                model.eval()
                torch.save(model.state_dict(), os.path.join(args.log_path, "epoch" + str(i) + '_model.pt'))


