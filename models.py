import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
from transformers import ViTModel

class ResNet50(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ViTb16(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_prob=0.3):
        super(ViTb16, self).__init__()

        if pretrained:
            self.base = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            self.base = ViTModel.from_pretrained(
                "google/vit-base-patch16-224", ignore_mismatched_sizes=True
            )

        hidden_size = self.base.config.hidden_size  # 768 for ViT-B/16

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)

        # CLS token embedding
        cls_token = outputs.last_hidden_state[:, 0]  # shape: [B, 768]

        x = self.dropout(cls_token)
        logits = self.classifier(x)

        return logits
