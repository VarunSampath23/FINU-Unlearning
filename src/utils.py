"""
src/utils.py

Utility functions for the Machine Unlearning project.
Includes device detection, random seed setting, and other helpers.
"""

import torch
import random
import numpy as np
import os


def get_device() -> torch.device:
    """
    Returns the best available device (CUDA > MPS > CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print(" Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print(" Using CPU (no GPU acceleration detected)")
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Additional settings for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility")


def create_dir_if_not_exists(path: str):
    """
    Create directory if it doesn't exist.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return total_params


def print_model_summary(model: torch.nn.Module, input_size=(3, 224, 224)):
    """
    Print a simple model summary (number of parameters + architecture).
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(model)
    count_parameters(model)
    print("="*60)


def save_checkpoint(model: torch.nn.Module, 
                    path: str, 
                    epoch: int = None, 
                    optimizer: torch.optim.Optimizer = None,
                    best_acc: float = None):
    """
    Save model checkpoint with optional metadata.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if best_acc is not None:
        checkpoint['best_acc'] = best_acc

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at: {path}")


def load_checkpoint(model: torch.nn.Module, 
                    path: str, 
                    device: torch.device = None, 
                    strict: bool = True):
    """
    Load model weights from checkpoint.
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Loaded model weights from {path}")
    else:
        model.load_state_dict(checkpoint, strict=strict)
        print(f"Loaded raw state dict from {path}")
    
    return model


# ====================== Optional: Progress Bar Helper ======================
def format_time(seconds: float) -> str:
    """Convert seconds to human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


if __name__ == "__main__":
    # Quick test
    set_seed(42)
    device = get_device()
    print(f"Device test: {device}")
