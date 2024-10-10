import torch
import torch.nn.functional as F

def focal_loss_with_logits(logits, targets, weight, alpha=0.25, gamma=2.0):
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean()


def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    BCE_loss = F.binary_cross_entropy(preds, targets, weight=weight)
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean()


def iou_loss(predicted, target):
    predicted = predicted.int()
    target = target.int()
    
    intersection = (predicted & target).sum().float()
    union = (predicted | target).sum().float()
    iou = intersection / union
    return iou