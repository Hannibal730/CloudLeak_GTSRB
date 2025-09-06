import torch.nn as nn
from torchvision import models
class VGG19DeepID(nn.Module):
    def __init__(self, num_classes=43, freeze_features=True):
        super().__init__()
        base=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features=base.features; self.avgpool=base.avgpool
        in_feat=25088
        self.deepid=nn.Sequential(nn.Linear(in_feat,480), nn.ReLU(inplace=True))
        self.fc7   =nn.Sequential(nn.Linear(480,4096), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.fc8   =nn.Linear(4096,num_classes)
        if freeze_features:
            for p in self.features.parameters(): p.requires_grad=False
    def forward(self,x,return_features=False):
        x=self.features(x); x=self.avgpool(x); x=x.view(x.size(0),-1)
        f=self.deepid(x); f7=self.fc7(f); logits=self.fc8(f7)
        if return_features: return logits, f7
        return logits
