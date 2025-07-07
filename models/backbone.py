import torch.nn as nn
import torchvision.models as models

class VisualBackbone(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            self.feature = nn.Sequential(*list(model.children())[:-1])
            self.out_dim = 2048
        else:
            raise NotImplementedError
    def forward(self, x):
        x = self.feature(x)
        return x.view(x.size(0), -1) 