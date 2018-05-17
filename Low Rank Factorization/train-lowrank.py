import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

import math

# Variables for training the model
args = {'arch': 'vgg19',
		  'workers': 4,
		  'epochs': 300,
		  'start_epoch': 0,
		  'batch_size': 128,
		  'learning_rate': 0.05,
		  'momentum': 0.9,
		  'weight_decay': 5e-4,
		  'print_freq': 20,
		  'resume': None,
		  'evaluate': False,
		  'half': False,
		  'save_dir': 'save_temp'}
best_prec1 = 0
cuda_enabled = torch.cuda.is_available()

def getLatestCheckpoint():
    lo = 0
    hi = args['epochs']
    while lo < hi:
        mid = int(math.ceil((lo+hi)/2))
        fname = os.path.join(args['save_dir'], 'checkpoint_{}.tar'.format(mid))
        if os.path.isfile(fname):
            lo = mid
        else:
            hi = mid-1
    if lo > 0 and lo == hi:
        return args['save_dir'] + '/checkpoint_' + str(lo) + '.tar'
    return None

def load_lowrank_model(path):
    global args
    global best_prec1
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        args['start_epoch'] = checkpoint['epoch']
        if 'best_prec1' in checkpoint:
            best_prec1 = checkpoint['best_prec1']
        lrm = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args['evaluate'], checkpoint['epoch']))
        if cuda_enabled:
            lrm.features = torch.nn.DataParallel(lrm.features)
            lrm.cuda()
        return lrm
    else:
        print("=> no checkpoint found at '{}'".format(path))

def train_and_evaluate_lowrank_model(lrm):
    global best_prec1

    if cuda_enabled:
        cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args['batch_size'], shuffle=True,
        num_workers=args['workers'], pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    if cuda_enabled:
        criterion.cuda()

    if args['half']:
        lrm.half()
        criterion.half()

    optimizer = torch.optim.SGD(lrm.parameters(), args['learning_rate'],
                                momentum=args['momentum'],
                                weight_decay=args['weight_decay'])

    if args['evaluate']:
        validate(val_loader, lrm, criterion)
        return

    for epoch in range(args['start_epoch'], args['epochs']):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_time, train_loss, train_acc = train(train_loader, lrm, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, val_loss = validate(val_loader, lrm, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': lrm.cpu(),
            'best_prec1': best_prec1,
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': prec1,
            'val_loss': val_loss
        }, is_best, filename=os.path.join(args['save_dir'], 'checkpoint_{}.tar'.format(epoch)))

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if cuda_enabled:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        if cuda_enabled:
            input_var = input_var.cuda()
        target_var = torch.autograd.Variable(target)
        if args['half']:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    return (batch_time.sum, losses.avg, top1.avg)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if cuda_enabled:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        if cuda_enabled:
            input_var = input_var.cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        if args['half']:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return (top1.avg, losses.avg)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args['learning_rate'] * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    model_path = getLatestCheckpoint()
    if model_path is None:
        model_path = os.path.join(args['save_dir'], 'lowrank-model.tar')
    lrm = load_lowrank_model(model_path)
    train_and_evaluate_lowrank_model(lrm)
