import torch.nn as nn
from torchvision import models
def get_resnet50(num_classes: int = 43, pretrained=True):
    m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    in_feats=m.fc.in_features; m.fc=nn.Linear(in_feats,num_classes); return m
