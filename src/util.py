import torch


def compute_accuracy(pred, target):
    acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
    return acc

def stack_graphs_with_padding(tsr, mask, batch_size):
    new_tsr = torch.zeros(mask.size(0), tsr.size(1)).to(tsr)
    new_tsr[mask] = tsr
    return new_tsr.reshape(batch_size, -1, tsr.size(1))
    