import torch.optim as optim
import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, List
from branch_reasoning.prompt_completion_dataset import PromptCompletion


def linear_warmup_decay(
    optimizer: optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Creates a schedule with a linear warmup and linear decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 0.1) / float(max(1, warmup_steps + 1))
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


def _print_gpu_memory():
    """Print detailed information about GPU memory usage."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    print("\nGPU Memory Information:")
    print("-" * 50)

    # Get number of GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")

    for i in range(n_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        # Get memory allocated
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # Convert to MB
        # Get memory reserved
        reserved = torch.cuda.memory_reserved(i) / 1024**2  # Convert to MB
        # Get total memory
        total = (
            torch.cuda.get_device_properties(i).total_memory / 1024**2
        )  # Convert to MB
        # Calculate free memory
        free = total - allocated

        print(f"Total Memory: {total:.2f} MB")
        print(f"Allocated Memory: {allocated:.2f} MB ({allocated/total*100:.1f}%)")
        print(f"Reserved Memory: {reserved:.2f} MB ({reserved/total*100:.1f}%)")
        print(f"Free Memory: {free:.2f} MB ({free/total*100:.1f}%)")

    print("-" * 50)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str | torch.device):
    """
    Moves all tensor states within the optimizer to the specified device.

    Args:
        optimizer: The PyTorch optimizer instance.
        device: The target device ('cpu', 'cuda', 'cuda:0', etc.) or torch.device object.
    """
    target_device = torch.device(device)  # Ensure it's a device object
    # Iterate over the state dictionary
    for state in optimizer.state.values():
        # Iterate over the parameter-specific state items (e.g., 'step', 'exp_avg')
        for k, v in state.items():
            # Check if the value is a tensor and not already on the target device
            if isinstance(v, torch.Tensor) and v.device != target_device:
                # Move the tensor to the target device
                state[k] = v.to(target_device)
    print(f"Optimizer states moved to {target_device}")


def find_largest_completion(prompt_completions: List[PromptCompletion]) -> int:
    max_length = 0
    for pc in prompt_completions:
        for bc in pc.branched_completions:
            for branch in bc.branches:
                if len(branch.completion) > max_length:
                    max_length = len(branch.completion)
    return max_length
