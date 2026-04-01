"""
metrics.py
Evaluation metrics and Membership Inference Attack (MIA) for machine unlearning experiments.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from typing import Dict


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a validation/test loader.
    Returns dictionary with Loss and Accuracy.
    """
    model.eval()
    outputs = []

    for batch in val_loader:
        if len(batch) == 3:  # For CifarSuper20 (image, fine, coarse)
            images, _, labels = batch
        else:
            images, labels = batch

        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)

        outputs.append({'Loss': loss.detach(), 'Acc': acc})

    # Aggregate
    batch_losses = [x['Loss'] for x in outputs]
    batch_accs = [x['Acc'] for x in outputs]

    return {
        'Loss': torch.stack(batch_losses).mean().item(),
        'Acc': torch.stack(batch_accs).mean().item()
    }


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of probability distribution.
    """
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=dim)


def collect_prob(loader: DataLoader, model: torch.nn.Module) -> torch.Tensor:
    """
    Collect softmax probabilities from the model over the entire loader.
    """
    model.eval()
    device = next(model.parameters()).device
    probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:  # CifarSuper20
                data, _, _ = batch
            else:
                data, _ = batch
            data = data.to(device, non_blocking=True)
            output = model(data)
            probs.append(F.softmax(output, dim=-1))

    return torch.cat(probs, dim=0)


def get_membership_attack_data(
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    model: torch.nn.Module
):
    """
    Prepare data for membership inference attack.
    Returns features and labels for retain vs test, and forget set.
    """
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    # Features: entropy of probabilities
    X_r = torch.cat([
        entropy(retain_prob),
        entropy(test_prob)
    ]).cpu().numpy().reshape(-1, 1)

    Y_r = np.concatenate([
        np.ones(len(retain_prob)),
        np.zeros(len(test_prob))
    ])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.ones(len(forget_prob))

    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    model: torch.nn.Module
) -> float:
    """
    Perform membership inference attack using Logistic Regression on entropy.
    Returns the fraction of forget samples predicted as members (MIA score).
    Lower is better for unlearning.
    """
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )

    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    clf.fit(X_r, Y_r)
    predictions = clf.predict(X_f)
    mia_score = predictions.mean()  # fraction predicted as "member"

    return float(mia_score)


# For quick testing
if __name__ == "__main__":
    print("Metrics module loaded successfully.")
    print("Available functions: accuracy, evaluate, get_membership_attack_prob")
