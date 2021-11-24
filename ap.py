from collections import defaultdict
from numpy.lib.function_base import average

import torch
import numpy as np


def distance(a, b):
    # the times 3 is to account for the fact that we normalized coords
    # by dividing by 3 in the data loader
    return 3 * (a - b).norm(p=2)


@torch.no_grad()
def compute_ap(groundtruth, prediction, thresholds=[]):
    groundtruth = groundtruth.cpu()
    prediction = prediction.cpu()
    for gt, pr in zip(groundtruth, prediction):
        gt = vec_to_properties(gt)
        pr = vec_to_properties(pr)
        gt = list(zip(*gt))
        pr = list(zip(*pr))

        properties_to_gt_coord = defaultdict(list)
        for m, c, p in gt:
            if m == 1:
                properties_to_gt_coord[p].append(c)

        tp = torch.zeros(len(thresholds), len(pr), dtype=torch.long)
        fp = torch.zeros(len(thresholds), len(pr), dtype=torch.long)

        # sort prediction on their confidences
        pr.sort(key=lambda x: x[0], reverse=True)
        for i, (mask, coord, properties) in enumerate(pr):
            candidates = properties_to_gt_coord[properties]
            # print(mask, coord, properties, candidates)
            best_dist, best_index = find_closest(coord, candidates)
            if best_index is None:
                fp[:, i] = True
                continue
            candidates.pop(best_index)
            for j, threshold in enumerate(thresholds):
                within_threshold = best_dist < threshold
                tp[j, i] = within_threshold
                fp[j, i] = not within_threshold
        acc_tp = tp.cumsum(1)
        acc_fp = fp.cumsum(1)
        precision = acc_tp / (acc_tp + acc_fp)
        recall = acc_tp / sum(x[0] for x in gt)
        
        results = []
        precision = precision.numpy()
        recall = recall.numpy()
        for p, r in zip(precision, recall):
            ap, *_ = CalculateAveragePrecision(r, p)
            results.append(ap)
        return torch.FloatTensor(results)


def find_closest(coord, list):
    best = float('inf')
    best_index = None
    for i, c in enumerate(list):
        dist = distance(coord, c)
        if dist < best:
            best = dist
            best_index = i
    return best, best_index


def vec_to_properties(vec):
    coord, material, color, shape, size, mask = vec.split((3, 2, 8, 3, 2, 1), dim=-1)
    properties = [material, color, shape, size]
    properties = [torch.argmax(x, dim=-1).tolist() for x in properties]
    properties = list(zip(*properties))
    return mask, coord, properties


# interpolation code copied from https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py#L294-L314
# by rafaelpaddila
def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0 : len(mpre) - 1], mrec[0 : len(mpre) - 1], ii]
