import numpy as np
import torch

#obs_rate = [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
obs_rate = [1/14,2/14,3/14,4/14,5/14,6/14,7/14,8/14,9/14,10/14,11/14,12/14,13/14,14/14]
num_rates = 14

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


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
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count
            
    def get_avg(self):
        return self.avg


def accuracy(output, target):

    maxk = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0)
    res = correct_k.mul_(100.0/ batch_size)
    return res