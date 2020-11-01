import torch
import torch.nn as nn


RESNET34_CONFIG = [3, 4, 6, 3]
RESNET18_CONFIG = [2, 2, 2, 2]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config, in_channels, normalize, out_dim, sobel, rot_head):
        super(ResNet, self).__init__()
        self.normalize = normalize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, config[0])
        self.layer2 = self._make_layer(BasicBlock, 64, 128, config[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, config[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, config[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.top_layer = nn.Linear(512, out_dim)
        if rot_head:
            self.rotnet_head = nn.Linear(512, 4)
        self._initialize_weights()
        self.sobel = self._create_sobel(sobel)

    def forward(self, x, rot_head=False, output_features=False):
        if self.sobel:
            x = self.sobel(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if rot_head:
            return self.rotnet_head(x)

        if output_features:
            return x

        x = self.top_layer(x)
        if self.normalize:
            x = torch.div(x, x.norm(2, dim=1).unsqueeze(1))

        return x

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _create_sobel(self, use_sobel):
        if use_sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
            return sobel
        return None


def resnet34(in_channels, rotnet_head, normalize=True, sobel=False, out_dim=10):
    """
    Creates a ResNet-34 for smaller images where the first layer has 3x3 filters and is followed by max pooling.
    :param in_channels: number of input channels
    :param rotnet_head: True/False add an additional head on ConvNet for rotation prediction
    :param normalize: True/False L2 normalize output of network
    :param sobel: True/False apply sobel filtering to input image
    :param out_dim: size of output layer
    :return: nn.Module
    """
    in_channels = 2 if sobel else in_channels
    return ResNet(config=RESNET34_CONFIG, in_channels=in_channels, normalize=normalize, out_dim=out_dim, sobel=sobel, rot_head=rotnet_head)


def resnet18(in_channels, rotnet_head, normalize=True, sobel=False, out_dim=10):
    """
    Creates a ResNet-18 for smaller images where the first layer has 3x3 filters and is followed by max pooling.
    :param in_channels: number of input channels
    :param rotnet_head: True/False add an additional head on ConvNet for rotation prediction
    :param normalize: True/False L2 normalize output of network
    :param sobel: True/False apply sobel filtering to input image
    :param out_dim: size of output layer
    :return: nn.Module
    """
    in_channels = 2 if sobel else in_channels
    return ResNet(config=RESNET18_CONFIG, in_channels=in_channels, normalize=normalize, out_dim=out_dim, sobel=sobel, rot_head=rotnet_head)
