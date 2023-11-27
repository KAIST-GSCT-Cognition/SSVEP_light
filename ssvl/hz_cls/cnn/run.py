import os
import argparse
import math
import random
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from ssvl.dataset import SSVEP_LIGHT
from ssvl.cnn.models.cnn1d import CNN1D
from ssvl.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams

parser = argparse.ArgumentParser(description='SSVEP')
parser.add_argument("--dataset_path", default="../../dataset", type=str)
parser.add_argument("--feature_type", default="waveform", type=str)
parser.add_argument("--platform", default="Sc", type=str)
parser.add_argument("--pid", default=None, type=str)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--warmup_steps', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--total_steps', default=3000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument("--cos", default=True, type=bool)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    tr_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type, platform=args.platform, pid=args.pid, split="train")
    # va_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type, platform=args.platform, pid=args.pid, split="valid")
    # te_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type,  platform=args.platform, pid=args.pid, split="test")
    train_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    model = CNN1D()
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    save_dir = f"exp/{args.platform}_{args.pid}"
    args.epochs = args.total_steps // len(train_loader)    
    print(args.epochs)

    logger = Logger(save_dir)
    save_hparams(args, save_dir)
    # best_val_loss = np.inf
    for epoch in range(0, args.epochs):
        train(train_loader, model, optimizer, epoch, logger, args)
        # if epoch % 10 == 0:
        #     torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/epoch{epoch}.pth')
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/last.pth')
    print("We are at epoch:", epoch)

def train(train_loader, model, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    cum_step = epoch * iters_per_epoch
    model.train()
    for data_iter_step, batch in enumerate(train_loader):
        x, y = batch
        lr = adjust_learning_rate(optimizer, cum_step + data_iter_step, args)
        x = x.cuda(args.gpu, non_blocking=True)
        y = y.cuda(args.gpu, non_blocking=True)
        # compute output
        loss = model(x, y)
        train_losses.step(loss.item(), x.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def adjust_learning_rate(optimizer, current_step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_steps = args.warmup_steps
    total_steps = args.total_steps
    if current_step < warmup_steps:
        lr = args.lr * current_step / warmup_steps
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()

    