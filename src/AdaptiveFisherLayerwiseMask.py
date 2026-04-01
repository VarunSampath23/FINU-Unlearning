import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class AdaptiveFisherLayerwiseMask:
    def __init__(self, model: nn.Module, device: torch.device, topk_min: float = 5.0, topk_max: float = 35.0,
                 exclude_bias: bool = True, exclude_norm: bool = True, soft_mask: bool = False):
        self.model = model
        self.device = device
        self.topk_min = topk_min
        self.topk_max = topk_max
        self.exclude_bias = exclude_bias
        self.exclude_norm = exclude_norm
        self.soft_mask = soft_mask
        self.criterion = nn.CrossEntropyLoss()

    def _skip_param(self, name: str) -> bool:
        if self.exclude_bias and "bias" in name:
            return True
        if self.exclude_norm and any(x in name.lower() for x in ["bn", "ln", "norm"]):
            return True
        return False

    def _zerolike_params_dict(self):
        return {k: torch.zeros_like(p, device=p.device) for k, p in self.model.named_parameters()}

    def _adaptive_topk_percent(self, layer_idx: int, total_layers: int) -> float:
        return self.topk_min + (self.topk_max - self.topk_min) * layer_idx / max(total_layers - 1, 1)

    @torch.no_grad()
    def _generate_masks(self, importances):
        masks = {}
        total_layers = len(importances)
        for idx, (name, imp) in enumerate(importances.items()):
            if self._skip_param(name):
                masks[name] = torch.zeros_like(imp)
                continue
            topk_percent = self._adaptive_topk_percent(idx, total_layers)
            flat = imp.view(-1)
            k = max(1, int(len(flat) * topk_percent / 100.0))
            thresh = torch.topk(flat, k, largest=True).values.min()
            if self.soft_mask:
                mask = torch.clamp((imp / (thresh + 1e-12)), 0.0, 1.0)
            else:
                mask = (imp >= thresh).float()
            masks[name] = mask
        return masks

    def compute_masks(self, dataloader: DataLoader):
        self.model.eval()
        importances = self._zerolike_params_dict()

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            for (name, param), imp in zip(self.model.named_parameters(), importances.values()):
                if param.grad is not None:
                    imp.add_(param.grad.detach().pow(2))

        for imp in importances.values():
            imp.div_(len(dataloader))

        return self._generate_masks(importances)