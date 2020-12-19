import torch

# from sklearn.metrics import auc, roc_auc_score, roc_curve


def compute_accuracy(logits: torch.Tensor, gold: torch.Tensor, sigfigs=3):
    total = len(logits)
    logits, pred = logits.max(1)
    right = (gold.eq(pred.long())).sum()
    return round(float(right) / float(total) * 100, sigfigs)


def compute_roc_score():
    # roc_auc_score
    pass
