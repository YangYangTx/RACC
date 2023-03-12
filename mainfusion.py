import os
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime

from torch.utils.data import Dataset
from torchvision import transforms, models
import functools
from torch.optim.lr_scheduler import OneCycleLR

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from swa_utils_fusion import AveragedModel, SWALR


from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate
from modelfusion import bqvsModelfusion
from datasetfusion import pbvsDatasetfusion
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'






parser = argparse.ArgumentParser(description='PyTorch  ImageNet  Classification')



parser.add_argument('--lr', default=1e-2, help='learing rate')
parser.add_argument('--batch_size', default=256, help='BS')
parser.add_argument('--epochs', default=45, help='EP')
parser.add_argument('--momentum', default=0.9, help='moment')
parser.add_argument('--weight_decay', default=5e-4, help='weight_decay')
parser.add_argument('--lr_steps', default=[10, 20, 30], help='lr_steps')
parser.add_argument('--val_freq', default=5, help='val')
parser.add_argument('--save_freq', default=1, help='val')
parser.add_argument('--iter_size', default=2, help='val')
parser.add_argument('--workers', default=12, help='val')
parser.add_argument('--print_freq', default=20, help='')
parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/")

parser.add_argument('--use_swa', type=int, default=1)

parser.add_argument('--input_size', default=224, help='input size')
parser.add_argument('--num_classes', default=10, help='choose model type')
parser.add_argument('--model_name_sa', default='resnet50', help='choose model type')
parser.add_argument('--model_name_eo', default='resnet50', help='choose model type')
parser.add_argument('--train_file', default='data/train_task_fusion.txt', help='train file path')
parser.add_argument('--val_file', default='data/val_task_fusion.txt', help='test file path')
parser.add_argument('--checkpoints_save_dir', default='checkpoints', help='save model')








best_prec1 = 0


def main():
    global best_prec1
    args = parser.parse_args()
    curtime = datetime.datetime.now()
    args.checkpoint_path = args.checkpoint_path  + "_".join([str(curtime.month), str(curtime.day), str(curtime.hour), str(curtime.minute)])
    print ("save dir is:", args.checkpoint_path)


    args.swa_epochs = 10
    model = bqvsModelfusion(args.model_name_sa, args.model_name_eo, args.num_classes, is_pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    print (model)  
   
    # use swa
    swa_model = AveragedModel(model.module)
    
  
    params = model.parameters()
    #params = filter(lambda p: p.requires_grad, model.parameters())


        # Data transforming
    clip_mean = [0.485, 0.456, 0.406]
    clip_std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=clip_mean,
                                     std=clip_std)


    train_transform = transforms.Compose([
        transforms.Resize(256),
		transforms.CenterCrop(args.input_size),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize])

    train_dataset = pbvsDatasetfusion(txt=args.train_file, transform=train_transform)
    val_dataset = pbvsDatasetfusion(txt=args.val_file, transform=val_transform)
    print('{} samples found, {} train samples and {} test samples.'.
          format(len(val_dataset) + len(train_dataset), len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    swa_loader = train_loader


    
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #
    optimizer = torch.optim.SGD(model.module.parameters(), lr=args.lr,  weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs + 7, steps_per_epoch=len(train_loader))
    swa_scheduler = SWALR(optimizer, swa_lr=3e-2, anneal_epochs=10)

   
    
    if not os.path.exists(args.checkpoints_save_dir):
        os.makedirs(args.checkpoints_save_dir)
    print("Saving everything to directory %s." % (args.checkpoints_save_dir))

    cudnn.benchmark = True

   
    saved_ckpt = []
    for epoch in range(0, args.epochs):
        args.epoch = epoch
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # use swa
        if args.use_swa and args.epoch >= args.swa_epochs:
            swa_model.update_parameters(model.module)
            swa_scheduler.step()


        # evaluate on validation set
        prec1 = 0.0
        if (epoch + 1) % args.val_freq == 0:
            prec1 = validate(val_loader, model, criterion, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save best Top3
        if len(saved_ckpt) < 3:
            saved_ckpt.append([prec1, 'epoch_' + str(epoch + 1)])
            path = args.checkpoint_path + 'epoch_' + str(epoch + 1)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), os.path.join(path, 'pytorch_model.pth'))
        else:
            if prec1 > saved_ckpt[-1][0]:
                
                shutil.rmtree(args.checkpoint_path + saved_ckpt[-1][1])
                saved_ckpt[-1] = [prec1, 'epoch_' + str(epoch + 1)]
                path = args.checkpoint_path + 'epoch_' + str(epoch + 1)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), os.path.join(path, 'pytorch_model.pth'))

        saved_ckpt.sort(key=lambda x: x[0], reverse=True)
        
        print('epoch: %s' % str(epoch + 1), saved_ckpt, flush=True)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model_name_sa,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_name, args.checkpoints_save_dir)
    
    # save swa model
    if args.use_swa:
        torch.optim.swa_utils.update_bn(swa_loader, swa_model, device=torch.device('cuda', 0))
        path = args.checkpoint_path
        torch.save(swa_model.module.state_dict(), os.path.join(path, 'swa_model.pth'))
        # test swa acc
        _ = validate(val_loader, swa_model, criterion, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    tic = time.time()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0

    for i, (input1, input2, target) in enumerate(train_loader):
        input1 = input1.float().cuda()
        input2 = input2.float().cuda()
        target = target.cuda()
        input_var1 = torch.autograd.Variable(input1)
        input_var2 = torch.autograd.Variable(input2)
        target_var = torch.autograd.Variable(target)
        #import pdb;pdb.set_trace()
        output = model(input_var1, input_var2)
        # measure accuracy and record loss
        pred = torch.max(output, 1)[1]
        train_correct = (pred == target).sum()
        acc_mini_batch += train_correct.item() * 100.0 / input1.size(0)
        loss = criterion(output, target_var)
        loss = loss / args.iter_size
        loss_mini_batch += loss.item()
        loss.backward()

        if (i + 1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss_mini_batch, input1.size(0))
            top1.update(acc_mini_batch / args.iter_size, input1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0

            if (i + 1) % args.print_freq == 0:
            #if (i + 1) % 90 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i + 1, len(train_loader) + 1, batch_time=batch_time, loss=losses, top1=top1))

    toc = time.time()

    print('Time used in this epoch: {}'.format(toc - tic))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    tic = time.time()

    end = time.time()
    for i, (input1, input2, target) in enumerate(val_loader):
        input1 = input1.float().cuda()
        input2 = input2.float().cuda()
        target = target.cuda()
        input_var1 = torch.autograd.Variable(input1)
        input_var2 = torch.autograd.Variable(input2)
        target_var = torch.autograd.Variable(target)

       
        # compute output
        output = model(input_var1, input_var2)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        pred = torch.max(output, 1)[1]
        num_correct = (pred == target).sum()
        prec1 = num_correct.item() * 100.0 / input1.size(0)
        losses.update(loss.item(), input1.size(0))
        top1.update(prec1, input1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    toc = time.time()
    print('Time used in this validation: {}'.format(toc - tic))

    return top1.avg





def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

