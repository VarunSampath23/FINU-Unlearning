import torch.nn as nn
from torchvision import models

def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "MobileNetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ViTb16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model