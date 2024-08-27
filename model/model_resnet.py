import sys

sys.path.append('/raid/sby/')
import torch
from torch import nn
import torchvision.models as models
class WeatherModel_out34(nn.Module):
    def __init__(self):
        super(WeatherModel_out34, self).__init__()
        net = models.resnet18(pretrained=True, progress=True)
        self.Resnet_conv1 = net.conv1
        self.Resnet_bn1 = net.bn1
        self.Resnet_relu = net.relu
        self.Resnet_maxpool = net.maxpool
        self.Resnet_layer1 = net.layer1
        self.Resnte_layer2 = net.layer2
        self.Resnet_avgpol = net.avgpool

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 8)

        self.relu = nn.ReLU()

        # self.output = nn.Softmax(dim=1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.Resnet_conv1(x)
        x = self.Resnet_bn1(x)
        x = self.Resnet_relu(x)
        x = self.Resnet_maxpool(x)
        x = self.Resnet_layer1(x)
        x = self.Resnte_layer2(x)
        x = self.Resnet_avgpol(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


model_out34 = WeatherModel_out34()



