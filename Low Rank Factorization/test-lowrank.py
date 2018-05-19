import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

args = {'workers': 4,
        'batch_size': 128,
        'print_freq': 20,
        'half': False,
        'save_dir': 'save_temp'}

cuda_enabled = torch.cuda.is_available()

def load_lowrank_model(path):
    global args
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        lrm = checkpoint['model']
        print("=> loaded checkpoint")
        if cuda_enabled:
            lrm.features = torch.nn.DataParallel(lrm.features)
            lrm.cuda()
        return lrm
    else:
        print("=> no checkpoint found at '{}'".format(path))

def test(loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(loader):
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
                      i, len(loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print('Accuracy: {top1.avg:.3f}'.format(top1=top1))
    print('Loss: {losses.avg:.3f}'.format(losses=losses))

    return (top1.avg, losses.avg)

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
    model_path = os.path.join(args['save_dir'], 'lowrank-model-best.tar')
    lrm = load_lowrank_model(model_path)
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                             shuffle=False, num_workers=args['workers'])
    test(testloader, lrm, nn.CrossEntropyLoss())