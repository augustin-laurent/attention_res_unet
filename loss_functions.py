import torch
from torch import Tensor

def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1):
    assert input.size() == target.size()
    #assert input.dim() == 3 or not reduce_batch_first

    #sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    target = _flatten(target)
    input = _flatten(input)

    inter = 2 * (input * target).sum(-1)
    sets_sum = input.sum() + target.sum(-1)
    #sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target, reduce_batch_first=True)