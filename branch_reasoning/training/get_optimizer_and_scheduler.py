import torch
import os
from typing import Dict, Any, Optional
from transformers import PreTrainedModel
from torch.optim.lr_scheduler import LambdaLR
from branch_reasoning.utils.utils import linear_warmup_decay


def get_optimizer_and_scheduler(
    model: PreTrainedModel,
    training_args: Dict[str, Any],
    iterations: int,
    no_completions: int,
    from_checkpoint: bool = False,
    optimizer_path: Optional[str] = None,
    scheduler_path: Optional[str] = None,
):
    # Create optimizer and scheduler outside the loop
    learning_rate = (
        training_args.get("learning_rate", 5e-6) if training_args is not None else 5e-6
    )
    weight_decay = (
        training_args.get("weight_decay", 0.00001)
        if training_args is not None
        else 0.00001
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_batch_size = training_args.get("batch_size", 1)
    total_steps_default = (
        iterations
        * training_args.get("num_epochs", 1)
        * (no_completions / train_batch_size)
    )
    total_steps = training_args.get("total_steps", total_steps_default)

    warmup_steps_default = max(1, int(total_steps * 0.1))  # 10% warmup
    warmup_steps = training_args.get("warmup_steps", warmup_steps_default)
    scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)

    if (
        from_checkpoint
        and optimizer_path is not None
        and os.path.exists(optimizer_path)
    ):
        optimizer.load_state_dict(torch.load(optimizer_path))
    if (
        from_checkpoint
        and scheduler_path is not None
        and os.path.exists(scheduler_path)
    ):
        scheduler.load_state_dict(torch.load(scheduler_path))

    return optimizer, scheduler


def save_optimizer_and_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    optimizer_path: str,
    scheduler_path: str,
):
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)
