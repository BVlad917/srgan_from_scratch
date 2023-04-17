from torch import nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        vgg_input_features = self.vgg(x)
        vgg_target_features = self.vgg(y)
        return self.loss(vgg_input_features, vgg_target_features)
