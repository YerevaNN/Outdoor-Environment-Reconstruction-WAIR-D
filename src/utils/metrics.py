# https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from torch import mean, Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def iou_score(pred: Tensor, target: Tensor, epsilon: float = 1e-7):
    # instead of calculating the IoU for the whole tensor(.sum())
    # we calculate for each batch(.sum(axis=[1, 2, 3])) and then average
    intersection = ((pred > 0.5) & (target.bool())).sum(axis=[1, 2, 3])
    union = ((pred > 0.5) | (target.bool())).sum(axis=[1, 2, 3])
    return mean((intersection + epsilon) / (union + epsilon))


def iog_score(pred: Tensor, target: Tensor, epsilon: float = 1e-7):
    """
    Intersection over ground truth. Something similar to recall.
    """
    intersection = ((pred > 0.5) & (target.bool())).sum(axis=[1, 2, 3])
    ground_truth = (target.bool()).sum(axis=[1, 2, 3])
    return mean((intersection + epsilon) / (ground_truth + epsilon))


def iop_score(pred: Tensor, target: Tensor, epsilon: float = 1e-7):
    """
    Intersection over prediction. Something similar to precision.
    """
    intersection = ((pred > 0.5) & (target.bool())).sum(axis=[1, 2, 3])
    pred_num = (pred > 0.5).sum(axis=[1, 2, 3])
    return mean((intersection + epsilon) / (pred_num + epsilon))


def hausdorff_dist(pred: Tensor, target: Tensor) -> float:
    hausdorff_metrics = []
    for pred_sample, supervision_sample in zip(pred, target):
        pred_idx = np.array(np.where(pred_sample[0].detach().cpu() > 0.5)).T
        supervision_idx = np.array(np.where(supervision_sample[0].detach().cpu() == 1)).T
        hausdorff_metrics.append(
            max(
                directed_hausdorff(pred_idx, supervision_idx)[0],
                directed_hausdorff(supervision_idx, pred_idx)[0]
            )
        )
    hausdorff_metric = float(np.mean(hausdorff_metrics))
    if hausdorff_metric == np.inf:
        return (pred.shape[-1] ** 2 + pred.shape[-2] ** 2) ** 0.5
    return hausdorff_metric


def chamfer_dist(pred: Tensor, target: Tensor) -> float:
    pred_idx = np.array(np.where(pred[0, 0] > 0.5)).T
    supervision_idx = np.array(np.where(target[0, 0] == 1)).T
    distances = ((pred_idx[np.newaxis] - supervision_idx[:, np.newaxis]) ** 2).sum(axis=-1) ** 0.5
    if distances.size == 0:
        return (pred.shape[-1] ** 2 + pred.shape[-2] ** 2) ** 0.5
    pred_dist = distances.min(axis=0)[0].mean()
    supervision_dist = distances.min(axis=1)[0].mean()
    return (pred_dist + supervision_dist) / 2
