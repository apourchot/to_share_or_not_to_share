import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.utils import _pair


class MaxPool(nn.Module):
    """
    MaxPool
    """
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x, *args):
        return self.op(x)


class ConvBNRelu(nn.Module):
    """
    Conv -> BN -> Relu
    """
    def __init__(self, c_in_max, c_out_max, kernel_size, stride, padding,
                 bn_momentum, bn_eps, conv_bias=False, bn_affine=True):

        super(ConvBNRelu, self).__init__()
        self.conv = VarConv2d(c_in_max, c_out_max, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        self.bn = VarBatchNorm2d(c_out_max, affine=bn_affine, momentum=bn_momentum, eps=bn_eps)
        self.relu = nn.ReLU(inplace=False)

        self.c_in_max = c_in_max
        self.c_out_max = c_out_max
        self.kernel_size = kernel_size

    def forward(self, x, c_out=None, kernel_size=None):

        if c_out is None:
            c_out = self.c_out_max
        if kernel_size is None:
            kernel_size = self.kernel_size

        N, C, H, W = x.size()

        out = self.conv(x, C, c_out, kernel_size)
        out = self.bn(out, c_in=c_out)
        out = self.relu(out)

        return out


class VarConv2d(Conv2d):
    """
    Conv2d with variable input and output size
    """
    def __init__(self, c_in_max, c_out_max, kernel_size_max, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size_max = _pair(kernel_size_max)
        stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
        super(VarConv2d, self).__init__(c_in_max, c_out_max, kernel_size_max, stride,
                                        padding, dilation, groups, bias)
        self.k_max = kernel_size_max

    def forward(self, x, c_in, c_out, kernel_size):
        w_tmp = self.weight[:c_out, :c_in,
                self.k_max[0] // 2 - kernel_size // 2:self.k_max[0] // 2 + kernel_size // 2 + 1,
                self.k_max[1] // 2 - kernel_size // 2:self.k_max[1] // 2 + kernel_size // 2 + 1]
        padding = kernel_size // 2
        return F.conv2d(x, w_tmp, None, self.stride,
                        padding, self.dilation, self.groups)


class VarBatchNorm2d(BatchNorm2d):
    """
    BatchNorm2d with variable input and output size
    """
    def __init__(self, c_in_max, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(VarBatchNorm2d, self).__init__(c_in_max, eps, momentum, affine,
                                             track_running_stats)

    def forward(self, x, c_in):
        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            x, self.running_mean[:c_in], self.running_var[:c_in], self.weight[:c_in] if self.affine else None,
            self.bias[:c_in] if self.affine else None, self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
