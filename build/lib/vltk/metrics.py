import torch

# from sklearn.metrics import auc, roc_auc_score, roc_curve


def accuracy(logits: torch.Tensor, gold: torch.Tensor, sigfigs=3):
    total = len(logits)
    logits, pred = logits.max(1)
    right = (gold.eq(pred.long())).sum()
    return round(float(right) / float(total) * 100, sigfigs)


def roc_score():
    # roc_auc_score
    pass


def soft_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1
