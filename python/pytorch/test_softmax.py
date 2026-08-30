import torch
from torch import Tensor

# softmax

def softmax(x: Tensor, dim: int=-1) -> Tensor:
    """softmax 

    Args:
        x (Tensor): _description_

    Returns:
        Tensor: _description_
    """
    # exp overflow
    # e^10 -> 22,026.4657948067
    # e^1000 -> ininity
    x_exp = x.exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def softmax_stable(x: Tensor, dim: int=-1) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_shift = x - x_max
    x_exp = x_shift.exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def test_softmax():
    # Large numbers in float16 will easily overflow 
    x = torch.randint(1000, 100000, (4, 100))
    out = softmax(x)
    
    # Check that the calculation broke (either inf OR nan)
    assert torch.any(torch.isinf(out)) or torch.any(torch.isnan(out)), "Expected unstable softmax to break!"
    
    # Alternative cleaner check: assert that NOT all elements are finite
    # assert not torch.all(torch.isfinite(out))


def test_softmax_stable():
    x = torch.randint(1000, 100000, (4, 100))
    out = softmax_stable(x)
        
    # Check that the calculation broke (either inf OR nan)
    assert not torch.any(torch.isinf(out)), "Expected unstable softmax to break!"
    assert not torch.any(torch.isnan(out)), "Expected unstable softmax to break!"
    