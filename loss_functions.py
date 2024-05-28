import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    #assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    dice = (2. * inter.sum() + epsilon) / (input.sum(dim=sum_dim) + target.sum(dim=sum_dim) + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)