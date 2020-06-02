import torch.nn as nn
from models.nasbench_201.utils import ReLUConvBN

class Zero(nn.Module):
    """
    zero op
    """
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Skip(nn.Module):
    """
    skip-connect op
    """

    def __init__(self):
        super(Skip, self).__init__()

    def forward(self, x):
        return x


class Conv1x1(nn.Module):
    """
    conv-1x1 op
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1,
                 conv_bias=False, bn_affine=True, bn_momentum=0.1, bn_eps=0.00001):
        super(Conv1x1, self).__init__()
        self.op = ReLUConvBN(in_channels, out_channels, 1, stride, padding, dilation,
                             conv_bias, bn_affine, bn_momentum, bn_eps, bn_eps)

    def forward(self, x):
        return self.op(x)


class Conv3x3(nn.Module):
    """
    conv-1x1 op
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=1, dilation=1,
                 conv_bias=False, bn_affine=True, bn_momentum=0.1, bn_eps=0.00001):
        super(Conv3x3, self).__init__()
        self.op = ReLUConvBN(in_channels, out_channels, 3, stride, padding, dilation,
                             conv_bias, bn_affine, bn_momentum, bn_eps, bn_eps)

    def forward(self, x):
        return self.op(x)


class AvgPool3x3(nn.Module):
    """
    avg-pool 3x3 op
    """
    def __init__(self):
        super(AvgPool3x3, self).__init__()
        self.op = nn.AvgPool2d(3, padding=1, stride=1)

    def forward(self, x, *args):
        return self.op(x)
