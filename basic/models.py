import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from basic.config import args


class LeNetZhuMNIST(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(588, num_classes)
        )
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

class Lenet5(nn.Module):
    def __init__(self, class_num=10):
        super(Lenet5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(args.image_channel, 6, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 84),
            nn.Sigmoid(),
            nn.Linear(84, class_num),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    pass
