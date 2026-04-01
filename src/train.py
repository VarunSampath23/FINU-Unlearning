"""
src/train.py

Training script for base models used in machine unlearning experiments.
Supports CIFAR-100, CIFAR Super-20, and ImageNet.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from src.models import get_model
from src.datasets import get_dataloaders, get_full_train_loader
from src.metrics import evaluate, accuracy
from src.utils import get_device, set_seed
from src.utils import get_device, set_seed, create_dir_if_not_exists, save_checkpoint


def training_step(model: nn.Module, batch, device: torch.device):
    """Single training step."""
    # For CIFAR Super-20: batch = (images, fine_label, coarse_label)
    # For others: batch = (images, label)
    if len(batch) == 3:          # Super-20 case
        images, _, labels = batch
    else:
        images, labels = batch

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    return loss


def fit_one_cycle(
    epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 0.0001,
    save_dir: str = None,
    model_name: str = "model",
    dataset_name: str = "cifar100"
) -> list:
    """
    Simple but effective training loop with ReduceLROnPlateau scheduler.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = []
    best_val_acc = 0.0
    best_path = None

    print(f"Starting training of {model_name} on {dataset_name} for {epochs} epochs...\n")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_losses = []

        # Training Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs} [Train]")
        for batch in pbar:
            loss = training_step(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        val_result = evaluate(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_result['Loss'])

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:2d}/{epochs}] | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_result['Loss']:.4f} | "
              f"Val Acc: {val_result['Acc']:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_result['Acc'] > best_val_acc:
            best_val_acc = val_result['Acc']
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, f"{model_name}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved! Val Acc: {best_val_acc:.2f}%")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_result['Loss'],
            'val_acc': val_result['Acc'],
            'lr': current_lr
        })

    # Save final model
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, f"{model_name}_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"\n Final model saved at: {final_path}")
        if best_path:
            print(f"Best model (Val Acc {best_val_acc:.2f}%) saved at: {best_path}")

    print(f"\n Training completed! Best Validation Accuracy: {best_val_acc:.2f}%\n")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train base model for unlearning experiments")
    
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cifar_super20", "imagenet"],
                        help="Dataset to train on")
    parser.add_argument("--model", type=str, default="ResNet18",
                        choices=["ResNet18", "MobileNetv2", "ViTb16"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Initial learning rate")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save models (default: Models/{dataset}/{model}/Training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

	
    

    # Prepare save directory
    if args.save_dir is None:
        args.save_dir = f"Models/{args.dataset}/{args.model}/Training"

    create_dir_if_not_exists(args.save_dir)

    print(f"Training {args.model} on {args.dataset.upper()} | Epochs: {args.epochs} | LR: {args.lr}")

    # Get full training and validation loaders (no forget/retain split)
    train_loader = get_full_train_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed
    )

    _, _, _, _, val_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        forget_type=None,           # None = full dataset mode
        batch_size=args.batch_size,
        seed=args.seed
    )

    # Create model
    model = get_model(model_name=args.model, num_classes=num_classes, pretrained=True)

    # Train
    history = fit_one_cycle(
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        save_dir=args.save_dir,
        model_name=args.model,
        dataset_name=args.dataset
    )

    print("Base model training finished successfully!")


if __name__ == "__main__":
    main()
