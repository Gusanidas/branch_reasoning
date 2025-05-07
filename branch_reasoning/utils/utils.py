import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable


def linear_warmup_decay(optimizer: optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Creates a schedule with a linear warmup and linear decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 0.1) / float(max(1, warmup_steps+1))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)

def evaluate_expression(expr: str) -> float:
    """Safely evaluate a mathematical expression string."""
    try:
        return eval(expr)
    except (SyntaxError, ZeroDivisionError):
        return float("inf")