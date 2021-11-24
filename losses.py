import numpy as np
import ray
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

def chamfer_loss(pred, target):
    """
    Args:
        pred: (batch, num_features, set_size)
        y: - // -
    """
    pdist = F.mse_loss(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1,),
        reduction='none').mean(-1)

    loss = pdist.min(1)[0] + pdist.min(2)[0]
    return loss.view(loss.size(0), -1).mean(1)


def l_split_ind(l, n):
    r = l%n
    return np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

@ray.remote
def lsa(arr, s, e):
    return np.array([linear_sum_assignment(p) for p in arr[s:e]])

def ray_lsa(arr, n):
    l = arr.shape[0]
    ind = l_split_ind(l, n)
    arr_id = ray.put(arr)
    res = [lsa.remote(arr_id, ind[i], ind[i+1]) for i in range(n)]
    res = np.concatenate([ray.get(r) for r in res])
    return res

def hungarian_loss(pred, target, num_workers=0):
    pdist = F.smooth_l1_loss(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    num_workers = min(pred.size(0), num_workers)
    if num_workers > 0:
        indices = ray_lsa(pdist_, num_workers)
    else:
        indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    total_loss = losses.mean(1)

    return total_loss


def hungarian_loss_numbering(inputs, pred, target, num_workers=0, ret_indices=False, loss_type='l2'):
    masks = 1e3 * (1 - torch.einsum("bsd,btd->bst", inputs, inputs))  # ensure matching inbetween classes

    if loss_type == 'ce':
        pred = pred.transpose(1,2)
        target = target.argmax(dim=2)
        pdist = F.cross_entropy(
            pred.unsqueeze(2).expand(-1, -1, target.size(1), -1), 
            target.unsqueeze(2).expand(-1, -1, pred.size(1)),
            reduction='none')
    elif loss_type == 'nl':
        print(pred.min(), pred.max())
        pred = torch.log(pred.transpose(1,2).clamp(min=1e-16))
        target = target.argmax(dim=2)
        pdist = F.nll_loss(
            pred.unsqueeze(2).expand(-1, -1, target.size(1), -1), 
            target.unsqueeze(2).expand(-1, -1, pred.size(1)),
            reduction='none')
    else:
        pdist = torch.cdist(target, pred, p=2.0)
    
    pdist = masks + pdist
    pdist_ = pdist.detach().cpu().numpy()

    num_workers = min(pred.size(0), num_workers)
    if num_workers > 0:
        indices = ray_lsa(pdist_, num_workers)
    else:
        indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    indices = torch.from_numpy(indices).to(device=pdist.device)
    losses = torch.gather(pdist.flatten(1,2), 1, indices)
    total_loss = losses.mean(1)

    if ret_indices:
        return total_loss, indices

    return total_loss


def hungarian_micro_accuracy(pred, target, indices):
    pred = pred.argmax(2)
    target = target.argmax(2)
    acc = pred.unsqueeze(1).expand(-1, target.size(1), -1) == target.unsqueeze(2).expand(-1, -1, pred.size(1))
    acc = torch.gather(acc.flatten(1,2), 1, indices)
    acc = acc.float().mean(1)
    return acc


def hungarian_macro_accuracy(pred, target, indices):
    pred = pred.argmax(2)
    target = target.argmax(2)
    acc = pred.unsqueeze(1).expand(-1, target.size(1), -1) == target.unsqueeze(2).expand(-1, -1, pred.size(1))
    acc = torch.gather(acc.flatten(1,2), 1, indices)
    acc = acc.all(1).float()
    return acc