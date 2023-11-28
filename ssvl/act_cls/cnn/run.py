import os
import argparse
import math
import random
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchmetrics import Accuracy
from ssvl.act_cls.dataset import SSVEP_LIGHT
from ssvl.models.cnn1d import CNN1D
from ssvl.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams

parser = argparse.ArgumentParser(description='SSVEP')
parser.add_argument("--dataset_path", default="../../../dataset", type=str)
parser.add_argument("--feature_type", default="waveform", type=str)
parser.add_argument("--platform", default="Sc", type=str)
parser.add_argument("--pid", default=None, type=str)
parser.add_argument("--target_hz", default="12hz", type=str)
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--warmup_steps', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--total_steps', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument("--cos", default=True, type=bool)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    tr_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type, platform=args.platform, pid=args.pid, hz=args.target_hz, split="train")
    va_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type, platform=args.platform, pid=args.pid, hz=args.target_hz, split="valid")
    te_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type,  platform=args.platform, pid=args.pid, hz=args.target_hz, split="test")
    eval_sets = torch.utils.data.ConcatDataset([va_dataset, te_dataset])
    train_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        eval_sets, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    model = CNN1D(n_class=2) # on or off
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    save_dir = f"exp/{args.platform}/{args.pid}_{args.target_hz}"
    args.epochs = args.total_steps // len(train_loader)  
    acc_metric = Accuracy(task="multiclass", num_classes=2)

    logger = Logger(save_dir)
    best_tr, best_val, early_stop = 0, 0, 0
    for epoch in range(0, args.epochs):
        train_acc = train(train_loader, model, optimizer, epoch, logger, acc_metric, args)
        val_acc = eval(test_loader, model, acc_metric, args)
        print(f"epoch: {epoch}, train acc: {train_acc}, val acc: {val_acc}")
        if val_acc > best_val:
            early_stop = 0
            best_val = val_acc
            best_tr = train_acc
            args.best_tr = float(best_tr)
            args.best_val = float(best_val)
            save_hparams(args, save_dir)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/best.pth')
        else:
            early_stop += 1
        if early_stop > 30:
            break
    print(f"finish best acc is {best_val}")

def train(train_loader, model, optimizer, epoch, logger, acc_metric, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    iters_per_epoch = len(train_loader)
    cum_step = epoch * iters_per_epoch
    model.train()
    avg_metric = []
    for data_iter_step, batch in enumerate(train_loader):
        x, y = batch
        lr = adjust_learning_rate(optimizer, cum_step + data_iter_step, args)
        x = x.cuda(args.gpu, non_blocking=True)
        y = y.cuda(args.gpu, non_blocking=True)
        acc_metric = acc_metric.cuda(args.gpu)
        # compute output
        loss = model(x, y)
        train_losses.step(loss.item(), x.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = model.inference(x)
        acc = acc_metric(pred, y)
        avg_metric.append(acc)
    return torch.mean(torch.stack(avg_metric))
            

def eval(test_loader, model, acc_metric, args):
    model.eval()
    test_pred, test_gt = [], []
    for data_iter_step, batch in enumerate(test_loader):
        x, y = batch
        x = x.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            predict = model.inference(x)
        test_pred.append(predict.detach().cpu())
        test_gt.append(y)
    preds = torch.cat((test_pred))
    target = torch.cat(test_gt)
    return acc_metric(preds, target)
    
        
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

    