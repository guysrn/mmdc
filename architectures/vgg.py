import torch
import torch.nn as nn

CONFIG = [32, 'M', 64, 'M', 128, 'M', 256, 'M']


class VGG(nn.Module):
    def __init__(self, config, in_size, in_channels, normalize, out_dim, sobel):
        super(VGG, self).__init__()
        self.features = self._make_layers(config=config, in_channels=in_channels)
        self.normalize = normalize
        self.top_layer = nn.Linear(1024, out_dim)
        self._initialize_weights()
        self.sobel = self._create_sobel(sobel)

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.top_layer(x)
        if self.normalize:
            x = torch.div(x, x.norm(2, dim=1).unsqueeze(1))
        return x

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

    def _make_layers(self, config, in_channels, batch_norm=True):
        layers = []
        in_channels = in_channels
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

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


def small_vgg(in_size, in_channels, normalize=True, sobel=False, batch_norm=True, out_dim=10):
    """
    Creates a small VGG-like ConvNet without the hidden fully connected layers.
    :param in_size: h/w of input image (assuming square)
    :param in_channels: number of input channels
    :param normalize: True/False L2 normalize output of network
    :param sobel: True/False apply sobel filtering to input image
    :param batch_norm: apply batch normalization after each conv layer
    :param out_dim: size of output layer
    :return: nn.Module
    """
    in_channels = 2 if sobel else in_channels
    return VGG(config=CONFIG,
               in_size=in_size,
               in_channels=in_channels,
               normalize=normalize,
               out_dim=out_dim,
               sobel=sobel)
