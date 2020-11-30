import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNets', 'resnet20', 'resnet34', 'resnet74', 'resnet146']


# Tensor shapes:
# Image Conv1 Block1 Block2 Block3
# 32x32 32x32 32x32  16x16  8x8
class ResNets(nn.Module):
    """
    Residual network implementation for CIFAR10 as described in paper [1].
    This implementation is hugely referenced to [2].

    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
    [2] https://github.com/akamaster/pytorch_resnet_cifar10
    """

    def __init__(self, block, block_shape, num_classes=10):
        super(ResNets, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.input_channels = 16
        self.block1 = self._create_block(block, 16, block_shape[0], stride=1)
        self.block2 = self._create_block(block, 32, block_shape[1], stride=2)
        self.block3 = self._create_block(block, 64, block_shape[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weight_init)

    def _create_block(self, block, channels, block_shape, first_stride):
        strides = [first_stride] + [1] * (block_shape - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_channels, channels, stride))
            self.input_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class BaseBlock(nn.Module):
    """
    Basic building block for residual networks
    """

    def __init__(self, in_channels, out_channels, stride, option='A'):
        super(BaseBlock, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.stride = stride
        # self.option = option

        # NN part
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual/shortcut part
        if stride == 1 and in_channels == out_channels:  # identical dimension with identity mapping
            self.shortcut = nn.Sequential()
        else:  # when dimension mismatch, there are option A and B in paper [1]
            if option == 'A':  # zero padding for increasing dimension
                self.shortcut = CustomLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  [0, 0, 0, 0, out_channels // 4, out_channels // 4]))
            if option == 'B':  # a project of x for increasing dimension, which is implemented by 1x1 conv, slightly
                # different from the paper
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                raise NotImplemented

    def foward(self, x):
        out = F.relu(self.b1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomLayer(nn.Module):
    def __init__(self, lambd):
        super(CustomLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def resnet20():
    return ResNets(BaseBlock, [3, 3, 3])


def resnet34():
    return ResNets(BaseBlock, [6, 6, 6])


def resnet74():
    return ResNets(BaseBlock, [12, 12, 12])


def resnet146():
    return ResNets(BaseBlock, [24, 24, 24])
