"""
src/unlearning.py

All unlearning methods for the FINU framework:
- FINU (Proposed)
- Random Labelling
- JiT  / Boundary Shrink (BDSH)
- Retraining baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms.v2 as v2

from src.AdaptiveFisherLayerwiseMask import AdaptiveFisherLayerwiseMask
from src.metrics import evaluate, get_membership_attack_prob
from src.utils import get_device


# ====================== FINU (Main Proposed Method) ======================
def learned_noise_unlearning(
    model: nn.Module,
    forget_dataloader: DataLoader,
    masks: dict = None,
    device: torch.device = None,
    epochs: int = 5,
    lr: float = 2e-5,
    delta_reg: float = 1.0,
    mask_scale: float = 1.0,
):
    """Learned additive noise on masked parameters (FUN core)."""
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    # Infer num_classes
    x_tmp, *_ = next(iter(forget_dataloader))
    x_tmp = x_tmp.to(device)
    num_classes = model(x_tmp).shape[-1]

    # Learnable noise parameters
    delta = {
        name: torch.zeros_like(param, device=device, requires_grad=True)
        for name, param in model.named_parameters()
    }

    if masks is None:
        masks = {name: torch.ones_like(param, device=device) for name, param in model.named_parameters()}

    optimizer = torch.optim.Adam(delta.values(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = []
        for batch in forget_dataloader:
            if len(batch) == 3:          # Super20 case
                x_f, _, y_f = batch
            else:
                x_f, y_f = batch

            x_f, y_f = x_f.to(device), y_f.to(device)
            optimizer.zero_grad()

            perturbed_weights = {
                k: model.state_dict()[k] + delta[k] * masks[k] * mask_scale
                for k in delta
            }

            outputs_f = torch.func.functional_call(model, perturbed_weights, x_f)

            reg_loss = sum((delta[k] ** 2).sum() for k in delta)
            loss = F.cross_entropy(outputs_f, y_f) - delta_reg * reg_loss

            (-loss).backward()          # Maximize loss on forget set
            optimizer.step()

            epoch_loss.append(loss.item())

        print(f"[FUN] Epoch {epoch+1}/{epochs} | Avg Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

    # Apply learned noise permanently
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(delta[name] * masks[name] * mask_scale)

    return model


# ====================== Random Labelling ======================
def random_labelling_unlearning(
    model: nn.Module,
    forget_dataloader: DataLoader,
    num_classes: int,
    epochs: int = 1,
    lr: float = 1e-3,
    device: torch.device = None,
):
    """Random relabelling baseline."""
    if device is None:
        device = get_device()

    model.train().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in forget_dataloader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch

            x, y = x.to(device), y.to(device)

            # Random relabel (exclude original label)
            y_rand = y.clone()
            for i in range(len(y_rand)):
                choices = list(range(num_classes))
                choices.remove(int(y[i].item()))
                y_rand[i] = random.choice(choices)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y_rand)
            loss.backward()
            optimizer.step()


# ====================== JiT / Lipschitz / Boundary Shrink (BDSH) ======================
class AddGaussianNoise:
    def __init__(self, mean=0., std=1., device='cpu'):
        self.mean = mean
        self.std = std
        self.device = device

    def __call__(self, tensor):
        _max = tensor.max()
        _min = tensor.min()
        tensor = tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean
        return torch.clamp(tensor, min=_min, max=_max)


class Lipschitz:
    def __init__(self, model, opt, device=None, parameters=None):
        self.model = model
        self.opt = opt
        self.device = device or get_device()
        self.n_samples = parameters.get("n_samples", 25)
        self.learning_rate = parameters.get("learning_rate", 0.001)
        self.lipschitz_weighting = parameters.get("lipschitz_weighting", 0.1)

        self.transforms = v2.Compose([
            AddGaussianNoise(0., self.lipschitz_weighting, device=self.device),
            v2.ToTensor(),
        ])

    def modify_weight(self, forget_dl: DataLoader):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for x, *_ in forget_dl:
            x = x.to(self.device)
            image = x.unsqueeze(0) if x.dim() == 3 else x
            out = self.model(image)

            loss = torch.tensor(0.0, device=self.device)
            for _ in range(self.n_samples):
                img2 = self.transforms(copy.deepcopy(x))
                image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2

                with torch.no_grad():
                    out2 = self.model(image2)

                flatimg = image.view(image.size(0), -1)
                flatimg2 = image2.view(image2.size(0), -1)

                in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)
                out_norm = torch.linalg.vector_norm(out - out2, dim=1)

                K = (out_norm / in_norm).sum().abs()
                loss += K

            loss /= self.n_samples
            loss.backward()
            optimizer.step()


def JiT(model, forget_train_dl, lipschitz_weighting=0.1, learning_rate=0.001):
    """JiT / Lipschitz unlearning wrapper."""
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "min_layer": -1,
        "max_layer": -1,
        "dampening_constant": 1,
        "selection_weighting": 10,
        "n_epochs": 5,
        "n_samples": 25,
        "learning_rate": learning_rate,
        "lipschitz_weighting": lipschitz_weighting,
        "use_quad_weight": False,
        "ewc_lambda": 1,
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    pdr = Lipschitz(model, optimizer, parameters=parameters)
    pdr.modify_weight(forget_train_dl)
    return model


# ====================== Boundary Shrink (BDSH) ======================
class AttackBase(object):
    def __init__(self, model=None, norm=False, discrete=True, device=None):
        self.model = model
        self.norm = norm
        # Normalization are needed for CIFAR10, ImageNet
        if self.norm:
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None):
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self, x):
        raise NotImplementedError

    def normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
            y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            return y
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255

    # Change this name as "projection"
    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                # TODO: channel isolation?
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)


class FGSM(AttackBase):
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, **kwargs):
        super(FGSM, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.rand = random_start

    # @overrides
    def perturb(self, x, y, model=None, bound=None, device=None, **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, x_nat, bound=bound,
                                 inverse_normalized=True)
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss":
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()


class LinfPGD(AttackBase):
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True,
                 device=None, **kwargs):
        super(LinfPGD, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.step = step
        self.iter = iters
        self.rand = random_start

    # @overrides
    def perturb(self, x, y, target_y=None, model=None, bound=None, step=None, iters=None, x_nat=None, device=None,
                **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        model.zero_grad()
        if x_nat is None:
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else:
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        for i in range(iters):
            adv_pred = model(x_adv)
            ori_pred = model(x)
            delta_pred = adv_pred - ori_pred
            if criterion.__class__.__name__ == "NLLLoss":
                delta_pred = F.log_softmax(delta_pred, dim=-1)
            # loss =   0.1*criterion(pred, target_y) - criterion(pred, original_y)
            if target_y is not None:
                # loss = criterion(adv_pred, y)
                loss = - criterion(delta_pred, target_y)  # + 0.01*criterion(delta_pred, y)
            else:
                loss = criterion(adv_pred, y)
            loss.backward()

            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
            
def boundary_shrink(
    ori_model,
    train_forget_loader,
    bound=0.1,
    poison_epoch=10,
    extra_exp=None,
    lambda_=0.7,
    bias=-0.5,
    slope=5.0,
):
    norm = True
    random_start = False

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)

    adv = FGSM(test_model, bound, norm, random_start, device)
    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=1e-5, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm(range(poison_epoch * batches_per_epoch)):

        x, _, y = next(forget_data_gen)
        x = x.to(device)
        y = y.to(device)

        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)

        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())

        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adversarial training
        unlearn_model.train()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        ori_loss = criterion(ori_logits, pred_label)

        if extra_exp == 'curv':
            ori_curv = curvature(ori_model, x, y, h=0.9)[1]
            cur_curv = curvature(unlearn_model, x, y, h=0.9)[1]
            delta_curv = torch.norm(ori_curv - cur_curv, p=2)
            loss = ori_loss + lambda_ * delta_curv

        elif extra_exp == 'weight_assign':
            weight = weight_assign(adv_logits, pred_label, bias=bias, slope=slope)
            loss = (
                torch.nn.functional.cross_entropy(
                    ori_logits, pred_label, reduction='none'
                ) * weight
            ).mean()
        else:
            loss = ori_loss

        loss.backward()
        optimizer.step()

    return unlearn_model

def bdsh(model, forget_train_dl, bound=0.1, poison_epoch=10):
    """Boundary Shrink / BDSH wrapper."""
    # Full implementation from your notebook can be added here
    # For now, placeholder that calls your original boundary_shrink
    from src.unlearning import boundary_shrink  # avoid circular import if needed
    model = boundary_shrink(model, forget_train_dl, bound=bound, poison_epoch=poison_epoch)
    return model


# ====================== Retraining Baseline ======================
def retrain_baseline(model, retain_train_dl, retain_test_dl, device, epochs=10, lr=0.0001):
    """Simple retraining on retain set only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(epochs):
        model.train()
        for batch in retain_train_dl:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        val_result = evaluate(model, retain_test_dl, device)
        scheduler.step(val_result['Loss'])
        print(f"Retrain Epoch {epoch+1} | Val Acc: {val_result['Acc']:.2f}%")

    return model


# ====================== Convenience Runner ======================
def run_unlearning_method(method: str, model, forget_train_dl, retain_train_dl, retain_test_dl,
                          num_classes, device, **kwargs):
    """Unified interface to run any unlearning method."""

    if method == "FINU":
        masker = AdaptiveFisherLayerwiseMask(model, device,
                                             topk_min=kwargs.get('topk_min', 5),
                                             topk_max=kwargs.get('topk_max', 35))
        masks = masker.compute_masks(forget_train_dl)
        return learned_noise_unlearning(model, forget_train_dl, masks=masks, device=device,
                                        epochs=kwargs.get('epochs', 50),
                                        lr=kwargs.get('lr', 2e-5),
                                        delta_reg=kwargs.get('delta_reg', 1.0))

    elif method == "random_labelling":
        random_labelling_unlearning(model, forget_train_dl, num_classes,
                                    epochs=kwargs.get('epochs', 5),
                                    lr=kwargs.get('lr', 1e-3),
                                    device=device)
        return model

    elif method == "JiT":
        return JiT(model, forget_train_dl,
                   lipschitz_weighting=kwargs.get('lipschitz_weighting', 0.1),
                   learning_rate=kwargs.get('lr', 0.001))

    elif method == "BDSH":
        return bdsh(model, forget_train_dl, bound=kwargs.get('bound', 0.1))

    elif method == "retrain":
        return retrain_baseline(model, retain_train_dl, retain_test_dl, device,
                                epochs=kwargs.get('epochs', 10),
                                lr=kwargs.get('lr', 0.0001))

    else:
        raise ValueError(f"Unknown unlearning method: {method}")
