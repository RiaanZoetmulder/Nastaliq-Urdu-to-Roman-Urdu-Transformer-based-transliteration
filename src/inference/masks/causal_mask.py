import torch


def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0


def batch_causal_mask(batch: int, size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(batch, 1, size, size), diagonal = 1).type(torch.int)
    return mask == 0
