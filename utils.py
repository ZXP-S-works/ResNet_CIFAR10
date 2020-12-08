import torch
import torch.nn as nn
from collections import defaultdict
from copy import copy


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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

    _, pred = output.topk(maxk, 1, True, True)  # ZXP ???
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class GradientNorm():
    def __init__(self, model):
        self.gradient_dict = defaultdict(float)
        self.gradient_hist = []
        self.model = model

    def update_hist(self):
        self.gradient_hist.append(list(self.gradient_dict.values()))
        self.gradient_dict = defaultdict(float)

    def calcu_gradient_norm(self):
        """
        updata gradient norm for each Conv2d layer
        """
        count = 0
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                for p in layer.parameters():
                    self.gradient_dict[str(name)] += p.detach().norm().item()

    def get_graidnet_norm_hist(self):
        return self.gradient_hist
