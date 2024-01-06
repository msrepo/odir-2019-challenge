from torchvision.models import resnet18
from torch import nn

N_CLASSES = 8
N_CHANNELS = 3
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
    
        self.model = resnet18(num_classes=N_CLASSES)

        self.model.conv1 = self.model.conv1 = nn.Conv2d(
            N_CHANNELS, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)