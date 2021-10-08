import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchinfo import summary

from utils.util import AverageMeter, ProgressMeter, accuracy, parse_gpus
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.data_process import get_folders
from models.caltech256 import create_net

def adjust_learning_rate(optimizer, epoch, args):
    # The initial LR decayed by 10 every 30 epochs
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    learning_rate = optimizer.param_groups[0]["lr"]

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))

    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.to(args.device, non_blocking=True)
        if torch.cuda.is_available():
            target = target.to(args.device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            epoch_msg = progress.get_message(i)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            print(epoch_msg)

        if i % args.log_freq == 0:
            args.log_file.write(epoch_msg + "\n")

def validate(val_loader, model, criterion, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
           
            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.to(args.device, non_blocking=True)

            # compute outputs
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                epoch_msg = progress.get_message(i)
                print(epoch_msg)

        epoch_msg = '----------- Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} -----------'.format(top1=top1, top5=top5)

        print(epoch_msg)

        args.log_file.write(epoch_msg + "\n")

    return top1.avg

def main(args, ngpus_per_node):

    if args.gpu is not None:
        assert(torch.cuda.is_available())
        cudnn.benchmark = True
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")

    # Data Loading
    print("Building dataset: Caltech256")

    args.num_class = 256
    train_folder, val_folder = get_folders(args.dataset_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_folder)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_folder, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_folder, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # create model
    model = create_net(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
    else:
        start_epoch = 0
        best_acc1 = 0

    ###########################################
    # thop
    # x = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(x,))
    # print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    # print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))
    # args.log_file.write("Params - " % str(params) + "\n")
    # args.log_file.write("FLOPs - " % str(flops) + "\n")
    ###########################################

    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Attention Module - " + args.attention_type + "\n")
    args.log_file.write("--------------------------------------------------" + "\n")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        if args.gpu is not None:
            torch.cuda.set_device(args.device)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        model = model.to(args.device) 

    with open(os.path.join(args.ckpt, "structure.txt"), mode="wt") as f:
        print(summary(model, input_size=(1, 3, 224, 224)), file=f)

    f.close()

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
        

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.cpu().state_dict(),
                "best_acc": best_acc1,
                "optimizer" : optimizer.state_dict(),
                }, is_best, epoch, save_path=args.ckpt)

        args.log_file.write("--------------------------------------------------" + "\n")

    args.log_file.write("best Acc@1 %4.2f" % best_acc1)

    print("Job Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Caltech256 Training Settings')

    # Model settings
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='model architecture (default: resnet18)')
    parser.add_argument("--attention_type", type=str, default="none",
                        help="attention type in building block (possible choices none|se|cbam|gc|eca|wa)")
    parser.add_argument("--attention_param", type=str, default="4",
                        help="attention parameter (reduction in (cbam/se/gc), kernel_size in eca(set 3), wavename in wa)")
    
    # Dataset settings
    parser.add_argument("--dataset_dir", type=str, default="",
                        help="data set path")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    
    # Optimizion settings
    parser.add_argument("--gpu", default="0",
                        help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    # DistributedDataParallel Settings
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # Misc
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument("--ckpt", default="./ckpts/", 
                        help="folder to output checkpoints")
    parser.add_argument("--log_freq", type=int, default=500,
                        help="log frequency to file")

    args = parser.parse_args()
    args.gpu = parse_gpus(args.gpu)

    args.ckpt += "caltech256"
    args.ckpt += "-" + args.arch
    if args.attention_type.lower() != "none":
        args.ckpt += "-" + args.attention_type
    if args.attention_type.lower() != "none":
        args.ckpt += "-param" + str(args.attention_param)

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    args.log_file = open(os.path.join(args.ckpt, "log_file.txt"), mode="w")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main(args, ngpus_per_node)

    args.log_file.close()