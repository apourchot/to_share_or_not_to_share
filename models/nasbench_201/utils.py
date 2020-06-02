import torch.nn as nn


class ReLUConvBN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 bias, affine, momentum=0.1, eps=0.00001, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(c_out, affine=affine, momentum=momentum, eps=eps, track_running_stats=track_running_stats)
        )

    def forward(self, x):
        return self.op(x)
  

class ResNetBasicBlock(nn.Module):

    def __init__(self, c_in, c_out, stride, affine=True):
        super(ResNetBasicBlock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(c_in, c_out, 3, stride, 1, 1, False, affine)
        self.conv_b = ReLUConvBN(c_out, c_out, 3, 1, 1, 1, False, affine)

        if stride == 2:
            self.down_sample = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                             nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False))
        elif c_in != c_out:
            self.down_sample = ReLUConvBN(c_in, c_out, 1, 1, 0, 1, False, affine)
        else:
            self.down_sample = None

        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__,
                                                                                **self.__dict__)
        return string

    def forward(self, inputs):

        basic_block = self.conv_a(inputs)
        basic_block = self.conv_b(basic_block)

        if self.down_sample is not None:
            residual = self.down_sample(inputs)
        else:
            residual = inputs
        return residual + basic_block
