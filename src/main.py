"""
src/main.py
Supports:
- Class unlearning
- Subclass unlearning  
- Sample unlearning
- Multiple methods: FINU, random_labelling, JiT, BDSH, retrain
"""

import argparse
import yaml
import os
import torch
import time

from src.models import get_model
from src.datasets import get_dataloaders
from src.unlearning import run_unlearning_method
from src.metrics import evaluate, get_membership_attack_prob
from src.utils import get_device, set_seed, create_dir_if_not_exists


def main():
    parser = argparse.ArgumentParser(description="FUN Machine Unlearning Framework")
    
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--method", type=str, default=None,
                        choices=["FUN", "random_labelling", "JiT", "BDSH", "retrain"],
                        help="Override unlearning method from config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Override config values from command line if provided
    if args.method:
        cfg["method"] = args.method
    if args.seed:
        cfg["seed"] = args.seed

    # Set reproducibility and device
    set_seed(cfg.get("seed", 42))
    device = get_device()

    print(f"\n{'='*90}")
    print(f" Starting {cfg['method']} unlearning on {cfg['dataset'].upper()}")
    print(f"Model: {cfg['model_name']} | Forget Type: {cfg['forget_type']}")
    if cfg['forget_type'] == 'class':
        print(f"Forget Class: {cfg.get('forget_class')}")
    elif cfg['forget_type'] == 'subclass':
        print(f"Forget Classes: {cfg.get('forget_classes')}")
    elif cfg['forget_type'] == 'sample':
        print(f"Forget Size: {cfg.get('forget_size')}")
    print(f"{'='*90}\n")

    start_total_time = time.time()

    # ====================== Load Data ======================
    forget_train_dl, retain_train_dl, forget_test_dl, retain_test_dl, test_dl, num_classes = \
        get_dataloaders(
            dataset_name=cfg["dataset"],
            forget_type=cfg["forget_type"],
            forget_class=cfg.get("forget_class"),
            forget_classes=cfg.get("forget_classes"),
            forget_size=cfg.get("forget_size"),
            batch_size=cfg.get("batch_size", 256),
            seed=cfg.get("seed", 42)
        )

    # ====================== Load Base Model ======================
    model = get_model(cfg["model_name"], num_classes=num_classes, pretrained=True).to(device)

    base_model_path = f"Models/{cfg['dataset']}/{cfg['model_name']}/Training/{cfg['model_name']}_final.pt"
    
    if os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print(f" Loaded pretrained base model from: {base_model_path}")
    else:
        print(f" Warning: Base model not found at {base_model_path}. Using randomly initialized model.")

    # ====================== Pre-Unlearning Evaluation ======================
    print("\nINITIAL (BEFORE UNLEARNING) EVALUATION:")
    print("-" * 60)
    
    pre_retain_test_result = evaluate(model, retain_test_dl, device)
    pre_forget_test_result = evaluate(model, forget_test_dl, device)
    
    pre_retain_train_result = evaluate(model, retain_train_dl, device)
    pre_forget_train_result = evaluate(model, forget_train_dl, device)

    pre_test_result = evaluate(model, test_dl, device)
    
    print(f"Retain Set Train Accuracy : {pre_retain_train_result['Acc']:.2f}%")
    print(f"Forget Set Train Accuracy : {pre_forget_train_result['Acc']:.2f}%")
    if cfg['forget_type'] == 'sample':
        print(f"Test Set Accuracy   : {pre_test_result['Acc']:.2f}%")
    else:
        print(f"Retain Set Accuracy : {pre_retain_test_result['Acc']:.2f}%")
        print(f"Forget Set Accuracy : {pre_forget_test_result['Acc']:.2f}%")
    
    pre_mia_score = get_membership_attack_prob(
        retain_train_dl, forget_train_dl, test_dl, model
    )
    print(f"Membership Inference Attack (MIA) Score : {pre_mia_score:.3f}\n")

    # ====================== Run Unlearning ======================
    method_kwargs = {
        "epochs": cfg.get("epochs", 50),
        "lr": cfg.get("lr", 2e-5),
        "topk_min": cfg.get("topk_min", 5.0),
        "topk_max": cfg.get("topk_max", 35.0),
        "delta_reg": cfg.get("delta_reg", 1.0),
        "lipschitz_weighting": cfg.get("lipschitz_weighting", 0.1),
        "bound": cfg.get("bound", 0.1),
    }

    model = run_unlearning_method(
        method=cfg["method"],
        model=model,
        forget_train_dl=forget_train_dl,
        retain_train_dl=retain_train_dl,
        retain_test_dl=retain_test_dl,
        num_classes=num_classes,
        device=device,
        **method_kwargs
    )

    total_time = time.time() - start_total_time

    # ====================== Final Evaluation ======================
    print("\nFINAL EVALUATION RESULTS:")
    print("-" * 60)

    retain_train_result = evaluate(model, retain_train_dl, device)
    forget_train_result = evaluate(model, forget_train_dl, device)
    
    retain_test_result = evaluate(model, retain_test_dl, device)
    forget_test_result = evaluate(model, forget_test_dl, device)
    test_result = evaluate(model, test_dl, device)

    print(f"Retain Set Train Accuracy : {retain_train_result['Acc']:.2f}%")
    print(f"Forget Set Train Accuracy : {forget_train_result['Acc']:.2f}%")
    if cfg['forget_type'] == 'sample':
        print(f"Test Set Accuracy   : {test_result['Acc']:.2f}%")
    else:
        print(f"Retain Set Test Accuracy : {retain_test_result['Acc']:.2f}%")
        print(f"Forget Set Test Accuracy : {forget_test_result['Acc']:.2f}%")
    print(f"Total Time          : {total_time:.1f} seconds")

    mia_score = get_membership_attack_prob(
        retain_train_dl, forget_train_dl, test_dl, model
    )
    print(f"Membership Inference Attack (MIA) Score : {mia_score:.3f}")

    # ====================== Save Results ======================
    output_dir = cfg.get("output_dir", f"results/{cfg['dataset']}_{cfg['forget_type']}_{cfg['method']}")
    create_dir_if_not_exists(output_dir)

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "unlearned_model.pt"))

    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Dataset          : {cfg['dataset']}\n")
        f.write(f"Model            : {cfg['model_name']}\n")
        f.write(f"Method           : {cfg['method']}\n")
        f.write(f"Forget Type      : {cfg['forget_type']}\n")
    
        if cfg['forget_type'] == 'class':
            f.write(f"Forget Class     : {cfg.get('forget_class')}\n")
        elif cfg['forget_type'] == 'sample':
            f.write(f"Forget Size      : {cfg.get('forget_size')}\n")
    
        # BEFORE
        f.write("\n--- BEFORE UNLEARNING ---\n")
        f.write(f"Retain Train Accuracy  : {pre_retain_train_result['Acc']:.2f}%\n")
        f.write(f"Forget Train Accuracy  : {pre_forget_train_result['Acc']:.2f}%\n")
        if cfg['forget_type'] == 'class':
            f.write(f"Retain Test Accuracy  : {pre_retain_test_result['Acc']:.2f}%\n")
            f.write(f"Forget Test Accuracy  : {pre_forget_test_result['Acc']:.2f}%\n")
        else:
            f.write(f"Test Accuracy    : {pre_test_result['Acc']:.2f}%\n")
        f.write(f"MIA Score        : {pre_mia_score:.3f}\n")
    
        # AFTER
        f.write("\n--- AFTER UNLEARNING ---\n")
        f.write(f"Retain Train Accuracy  : {retain_train_result['Acc']:.2f}%\n")
        f.write(f"Forget Train Accuracy  : {forget_train_result['Acc']:.2f}%\n")
        if cfg['forget_type'] == 'class':
            f.write(f"Retain Test Accuracy  : {retain_test_result['Acc']:.2f}%\n")
            f.write(f"Forget Test Accuracy  : {forget_test_result['Acc']:.2f}%\n")
        else:
            f.write(f"Test Accuracy    : {test_result['Acc']:.2f}%\n")
        f.write(f"MIA Score        : {mia_score:.3f}\n")
        f.write(f"\nTotal Time (s)   : {total_time:.1f}\n")

    print(f"\nExperiment completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
