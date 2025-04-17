# This is the model architecture. Note that it is not used in the actual engine, since the model has been converted to ONNX format.
from helperfuncs import *

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class ComplexModel(nn.Module):
  # Attempt at using a deeper CNN.

    def __init__(self, name):
        super().__init__()

        self.name = name

        self.conv_net = nn.Sequential()
        self.conv_net.add_module("Conv 1", nn.Conv2d(1, 200, 3, 1, 1))
        self.conv_net.add_module("SE 1", SE_Block(200))
        self.conv_net.add_module("Batchnorm 1", nn.BatchNorm2d(200))
        self.conv_net.add_module("Conv activation", nn.Mish())
        self.conv_net.add_module("Conv 2", nn.Conv2d(200, 190, 3, 1, 1))
        self.conv_net.add_module("SE 2", SE_Block(190))
        self.conv_net.add_module("Batchnorm 2", nn.BatchNorm2d(190))
        self.conv_net.add_module("Conv activation 2", nn.Mish())
        self.conv_net.add_module("Conv 3", nn.Conv2d(190, 180, 3, 1, 1))
        self.conv_net.add_module("SE 3", SE_Block(180))
        self.conv_net.add_module("Batchnorm 3", nn.BatchNorm2d(180))
        self.conv_net.add_module("Conv activation 3", nn.Mish())
        self.conv_net.add_module("Conv 4", nn.Conv2d(180, 170, 3, 1, 1))
        self.conv_net.add_module("SE 4", SE_Block(170))
        self.conv_net.add_module("Batchnorm 4", nn.BatchNorm2d(170))
        self.conv_net.add_module("Conv activation 4", nn.Mish())
        self.conv_net.add_module("Conv 5", nn.Conv2d(170, 160, 3, 1, 1))
        self.conv_net.add_module("SE 5", SE_Block(160))
        self.conv_net.add_module("Batchnorm 5", nn.BatchNorm2d(160))
        self.conv_net.add_module("Conv activation 5", nn.Mish())
        self.conv_net.add_module("Conv 6", nn.Conv2d(160, 150, 3, 1, 1))
        self.conv_net.add_module("SE 6", SE_Block(150))
        self.conv_net.add_module("Batchnorm 6", nn.BatchNorm2d(150))
        self.conv_net.add_module("Conv activation 6", nn.Mish())
        self.conv_net.add_module("Conv 7", nn.Conv2d(150, 140, 3, 1, 1))
        self.conv_net.add_module("SE 7", SE_Block(140))
        self.conv_net.add_module("Batchnorm 7", nn.BatchNorm2d(140))
        self.conv_net.add_module("Conv activation 7", nn.Mish())
        self.conv_net.add_module("Conv 8", nn.Conv2d(140, 130, 3, 1, 1))
        self.conv_net.add_module("SE 8", SE_Block(130))
        self.conv_net.add_module("Batchnorm 8", nn.BatchNorm2d(130))
        self.conv_net.add_module("Conv activation 8", nn.Mish())
        self.conv_net.add_module("Conv 9", nn.Conv2d(130, 120, 3, 1, 1))
        self.conv_net.add_module("SE 9", SE_Block(120))
        self.conv_net.add_module("Batchnorm 9", nn.BatchNorm2d(120))
        self.conv_net.add_module("Conv activation 9", nn.Mish())
        self.conv_net.add_module("Conv 10", nn.Conv2d(120, 110, 3, 1, 1))
        self.conv_net.add_module("SE 10", SE_Block(110))
        self.conv_net.add_module("Batchnorm 10", nn.BatchNorm2d(110))
        self.conv_net.add_module("Conv activation 10", nn.Mish())
        self.conv_net.add_module("Flattener", nn.Flatten())

        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear 1", nn.Linear(7040, 1))
        self.mlp.add_module("Activation 1", nn.Sigmoid())

    def forward(self, x):
        a=self.conv_net(x)
        return self.mlp(self.conv_net(x))

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)