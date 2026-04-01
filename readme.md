# FINU: Fisher-Informed Noise Injection for Efficient Zero-Shot Unlearning

This code is official implementation for the paper *"FINU: Fisher-Informed Noise Injection for Efficient Zero-Shot Unlearning"*.

By Varun Sampath Kumar, Esmaeil S Nadimi, Vinay Chakravarthi Gogineni

**FINU** = Adaptive Fisher Layerwise Mask + Learned Additive Noise (the proposed method).

Supports:
- Class-level unlearning (CIFAR-100, ImageNet)
- Subclass unlearning (CIFAR Super-20)
- Sample-level (instance) unlearning (CIFAR-100)

Models supported:
- ResNet18
- ViTB-16

Metrics Supported:
- Accuracy
- Membership Inference Attack (MIA)

Methods Supported:
- Retraining
- Random Labelling
- BDSH [Chen et al CVPR 2023]
- JiT [Jack Foster et al TMLR 2025]
- FINU [Proposed method]





