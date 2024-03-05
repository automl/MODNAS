##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ["avg_pool_3x3","nor_conv_1x1","nor_conv_3x3","none","skip_connect"]
OPS = {
    "none":
    lambda C_in, C_out, stride, affine, track_running_stats: Zero(
        C_in, C_out, stride),
    "avg_pool_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats),
    "max_pool_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "max", affine, track_running_stats),
    "nor_conv_3x3":
    lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1":
    lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "skip_connect":
    lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if stride == 1 and C_in == C_out else FactorizedReduce(
        C_in, C_out, stride, affine, track_running_stats),
}

NAS_BENCH_201 = [
    "none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"
]


class SubConv(nn.Module):

    def __init__(self, op, kernel_size):
        super(SubConv, self).__init__()
        #print(op.weight.shape)
        self.kernel_size = kernel_size
        self.op = op

    def forward(self, x):
        new_padding = int(
            (self.kernel_size - 1) / 2)  # Assumes stride = 1, dilation = 1
        x = F.conv2d(x,
                     weight=self.op.weight[:, :, 1:(self.kernel_size + 1),
                                           1:(self.kernel_size + 1)],
                     bias=self.op.bias,
                     stride=self.op.stride,
                     padding=new_padding,
                     groups=self.op.groups)
        return x


class ReLUConvBNSubSample(nn.Module):

    def __init__(self, layer, kernel_size):
        super(ReLUConvBNSubSample, self).__init__()

        self.op = nn.Sequential(
            layer.op[0],
            SubConv(layer.op[1], kernel_size),
            layer.op[2],
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(C_out,
                           affine=affine,
                           track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)

class ReLUConvBNMixture(nn.Module):
    def __init__(self, op, kernel_sizes):
        super(ReLUConvBNMixture, self).__init__()
        self.op = op

        assert len(kernel_sizes) == 2 # Assuming only 2 operations are entangled for now

        self.kernel_list = kernel_sizes
        self.kernel_max = max(kernel_sizes)

    def _compute_weight_and_bias(self, weights, idx, conv_weight, conv_bias, use_argmax=False):
        alpha = weights[idx]

        kernel_size = self.kernel_list[idx]
        start = 0 + (self.kernel_max - kernel_size) // 2
        end = start + kernel_size
        weight_curr = self.op.op[1].weight[:, :, start:end, start:end]
        if use_argmax == True:
            conv_weight += alpha * weight_curr
        else:
            conv_weight += alpha * F.pad(weight_curr, (start, start, start, start), "constant", 0)

        if self.op.op[1].bias is not None:
            conv_bias = self.op.op[1].bias

        return conv_weight, conv_bias
    
    def get_padding(self, kernel_size):
        if kernel_size == 3:
            return 1    
        else:
            return 0

    def forward(self, x, weights, use_argmax=False):
        x = self.op.op[0](x)

        conv_weight = 0
        conv_bias = 0

        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            conv_weight, conv_bias = self._compute_weight_and_bias(
                weights=weights,
                idx=argmax,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                use_argmax=use_argmax
            )
            selected_kernel_size = self.kernel_list[argmax]
        else:
            for i, _ in enumerate(weights):
                conv_weight, conv_bias = self._compute_weight_and_bias(
                    weights=weights,
                    idx=i,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    use_argmax=use_argmax
                )
            selected_kernel_size = self.kernel_max

        conv_bias = conv_bias if isinstance(conv_bias, torch.Tensor) else None

        x = F.conv2d(x,
                weight=conv_weight,
                bias=conv_bias,
                stride=self.op.op[1].stride,
                padding=self.get_padding(selected_kernel_size),
                dilation = self.op.op[1].dilation,
                groups = self.op.op[1].groups)

        x = self.op.op[2](x)
        return x

class ResNetBasicblock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 affine=True,
                 track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine,
                                 track_running_stats)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine,
                                 track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine,
                                         track_running_stats)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 stride,
                 mode,
                 affine=True,
                 track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine,
                                         track_running_stats)
        if mode == "avg":
            self.op = nn.AvgPool2d(3,
                                   stride=stride,
                                   padding=1,
                                   count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(
            **self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in,
                              C_outs[i],
                              1,
                              stride=stride,
                              padding=0,
                              bias=not affine))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in,
                                  C_out,
                                  1,
                                  stride=stride,
                                  padding=0,
                                  bias=not affine)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(C_out,
                                 affine=affine,
                                 track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])],
                            dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(
            **self.__dict__)
