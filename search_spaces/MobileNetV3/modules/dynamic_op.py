# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import itertools
from search_spaces.MobileNetV3.utils import (
    get_same_padding,
    sub_filter_start_end,
    make_divisible,
    SEModule,
    MyNetwork,
    MyConv2d,
)

__all__ = [
    "DynamicSeparableConv2d",
    "DynamicConv2d",
    "DynamicBatchNorm2d",
    "DynamicGroupNorm",
    "DynamicSE",
    "DynamicLinear",
    "DynamicInConv2dMixture",
    "DynamicOutConv2dMixture"
]


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small ** 2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel,
                                   :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(x, filters.to(x.device), None, self.stride,
                     padding, self.dilation, in_channel)
        return y


class DynamicInSeparableConv2dMixture(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, in_channels_list, kernel_size_list, stride=1, dilation=1, use_we_v2=False):
        super(DynamicInSeparableConv2dMixture, self).__init__()

        self.max_in_channels = max(in_channels_list)
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.in_channels_list = in_channels_list
        self.max_kernel_size = max(kernel_size_list)
        self.use_we_v2 = use_we_v2
        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small ** 2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size, use_argmax=False):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        #print(self.conv.weight.shape)
        filters = self.conv.weight[:out_channel,
                                   :in_channel, start:end, start:end]
        # filters = F.pad(filters, (0, self.max_kernel_size-kernel_size, 0 ,self.max_kernel_size-kernel_size, 0, self.max_in_channels-in_channel,0,self.max_in_channels-in_channel), "constant", 0)
        #print(filters.shape)
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        padding = ((self.max_kernel_size)-kernel_size)//2
        #print(filters.shape)
        if use_argmax:
            return filters
        filters = F.pad(filters, (padding, padding, padding, padding, 0, 0, 0, self.max_in_channels-in_channel), "constant", 0)
        return filters

    def get_weights_mixture(self, weights, use_argmax=False):
        channels_kernel_cross = list(itertools.product(
            self.in_channels_list, self.kernel_size_list))
        conv_weight = 0
        if use_argmax:
            argmax = np.array([w.item() for w in weights]).argmax()
            in_channels, kernel_size = channels_kernel_cross[argmax]
            if self.use_we_v2:
                conv_weight = self.get_active_filter(in_channels, kernel_size,use_argmax=use_argmax) * weights[argmax]
            else:
                conv_weight = self.get_active_filter(in_channels, kernel_size,use_argmax=use_argmax)
        else:
            for i, w in enumerate(weights):
                in_channels, kernel_size = channels_kernel_cross[i]
                #print(in_channels)
                #print(kernel_size)
                conv_weight = conv_weight + w * \
                    self.get_active_filter(in_channels, kernel_size)
        #print(conv_weight)

        return conv_weight

    def forward(self, x, weights, use_argmax=False, kernel_size=None):
        #print("DynamicInSeparableConv2dMixture using we_v2: ", self.use_we_v2)
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        filters = self.get_weights_mixture(
            weights, use_argmax=use_argmax).contiguous()
        channels_kernel_cross = list(itertools.product(
            self.in_channels_list, self.kernel_size_list))
        padding = get_same_padding(max(self.kernel_size_list))
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        if use_argmax:
            argmax = np.array([w.item() for w in weights]).argmax()
            in_channels, kernel_size = channels_kernel_cross[argmax]
            padding = get_same_padding(kernel_size)
            #print(x.shape)
            #print(in_channels)
            if self.use_we_v2:
                y = F.conv2d(x[:, :in_channels, :, :], filters, None, self.stride, padding, self.dilation, in_channels)
            else:
                y = F.conv2d(x[:, :in_channels, :, :], filters, None, self.stride, padding, self.dilation, in_channels)*weights[argmax]
        else:
            y = F.conv2d(x, filters.to(x.device), None, self.stride, padding,
                     self.dilation, self.max_in_channels)
        return y


class DynamicConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        # MODIF : Add mixture of filters here
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicInConv2dMixture(nn.Module):
    def __init__(
        self, in_channels_list, out_channels, kernel_size=1, stride=1, dilation=1, use_we_v2=False
    ):
        super(DynamicInConv2dMixture, self).__init__()

        self.in_channels_list = in_channels_list
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.use_we_v2 = use_we_v2

        self.conv = nn.Conv2d(
            max(self.in_channels_list),
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

    def get_active_filter(self, out_channel, in_channel, use_argmax=False):
        # MODIF : Add mixture of filters here
        if use_argmax:
            if self.use_we_v2:
                return self.conv.weight[:out_channel, :in_channel, :, :]
            return self.conv.weight[:out_channel, :in_channel, :, :]
        return F.pad(self.conv.weight[:out_channel, :in_channel, :, :], (0, 0, 0, 0, 0, max(self.in_channels_list)-in_channel, 0, 0), "constant", 0)

    def get_filter_mixture(self, weights, use_argmax=False):
        conv_weight = 0
        channels = self.in_channels_list
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            in_channels = channels[argmax]
            if self.use_we_v2:
                conv_weight = self.get_active_filter(
                self.out_channels, in_channels,use_argmax=use_argmax) * weights[argmax]
            else:
                conv_weight = self.get_active_filter(
                self.out_channels, in_channels,use_argmax=use_argmax)
        else:
            for i, w in enumerate(weights):
                in_channels = channels[i]
                conv_weight = conv_weight + w * \
                    self.get_active_filter(self.out_channels, in_channels)

        return conv_weight

    def forward(self, x, weights, use_argmax=False):
        #print("DynamicInConv2dMixture using we_v2: ", self.use_we_v2)
        filters = self.get_filter_mixture(
            weights, use_argmax=use_argmax).contiguous()
        padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        if use_argmax:
            argmax = np.array([w.item() for w in weights]).argmax()
            in_channels = self.in_channels_list[argmax]
            if self.use_we_v2:
               y = F.conv2d(x[:, :in_channels, :, :], filters.to(x.device), None, self.stride, padding, self.dilation, 1)
            else:
                y = F.conv2d(x[:, :in_channels, :, :], filters.to(x.device), None, self.stride, padding, self.dilation, 1)*weights[argmax]
        else:
            y = F.conv2d(x, filters.to(x.device), None, self.stride, padding, self.dilation, 1)
        return y


class DynamicOutConv2dMixture(nn.Module):
    def __init__(
        self, in_channels, out_channels_list, kernel_size=1, stride=1, dilation=1, use_we_v2=False
    ):
        super(DynamicOutConv2dMixture, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channel_list = out_channels_list
        self.stride = stride
        self.dilation = dilation
        self.use_we_v2 = use_we_v2

        self.conv = nn.Conv2d(
            self.in_channels,
            max(self.out_channel_list),
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

    def get_active_filter(self, out_channel, in_channel, use_argmax=False):
        # MODIF : Add mixture of filters here
        filters = self.conv.weight[:out_channel, :in_channel, :, :]
        #print(filters)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        if use_argmax:
            #print(filters)
            return filters
        return F.pad(filters, (0, 0, 0, 0, 0, 0, 0, max(self.out_channel_list)-out_channel), "constant", 0)

    def get_filter_mixture(self, weights, use_argmax=False):
        conv_weight = 0
        channels = self.out_channel_list
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            out_channels = channels[argmax]
            if self.use_we_v2:
               conv_weight = self.get_active_filter(
                out_channels, self.in_channels, use_argmax=use_argmax) * weights[argmax]
            else:
                conv_weight = self.get_active_filter(
                out_channels, self.in_channels, use_argmax=use_argmax)
        else:
            for i, w in enumerate(weights):
                out_channels = channels[i]
                conv_weight = conv_weight + w * \
                    self.get_active_filter(out_channels, self.in_channels)

        return conv_weight

    def forward(self, x, weights, use_argmax=False):
        #print("DynamicOutConv2dMixture using we_v2: ", self.use_we_v2)
        filters = self.get_filter_mixture(
            weights, use_argmax=use_argmax).contiguous()
        #print(filters)
        padding = get_same_padding(self.kernel_size)
        if use_argmax:
         if self.use_we_v2:
           y = F.conv2d(x, filters.to(x.device), None, self.stride, padding, self.dilation, 1)
         else:
           argmax = np.array([w.item() for w in weights]).argmax()
           y = F.conv2d(x, filters.to(x.device), None, self.stride, padding, self.dilation, 1)*weights[argmax]
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        # MODIF: add bn mixture here
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / \
                            float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicBatchNorm2dMixture(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, feature_dim_list, use_we_v2=False):
        super(DynamicBatchNorm2dMixture, self).__init__()

        self.feature_dim_list = feature_dim_list
        self.max_feature_dim = max(self.feature_dim_list)
        self.bn = nn.BatchNorm2d(self.max_feature_dim)
        self.use_we_v2 = use_we_v2

    def get_active_mean_var(self, bn, out_features):
        mean = F.pad(bn.running_mean[:out_features], (0, max(
            self.feature_dim_list)-out_features), "constant", 0)
        var = F.pad(bn.running_var[:out_features], (0, max(
            self.feature_dim_list)-out_features), "constant", 0)
        return mean, var

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim_list, feature_dim, weights, use_argmax=False, use_we_v2=False):
      def get_active_weights_bias(bn, out_features, feature_dim_list, use_argmax=False):
        if use_argmax:
            return bn.weight[:out_features], bn.bias[:out_features]
        weight = F.pad(bn.weight[:out_features], (0, max(
            feature_dim_list)-out_features), "constant", 0)
        bias = F.pad(bn.bias[:out_features], (0, max(
            feature_dim_list)-out_features), "constant", 0)
        return weight, bias

      def get_params_mixture(bn, weights, feature_dim_list, use_argmax=False, use_we_v2=False):
        weight = 0
        bias = 0
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            n_channels = feature_dim_list[argmax]
            w, b = get_active_weights_bias(bn, n_channels, feature_dim_list, use_argmax=use_argmax)
            if use_we_v2:
                weight, bias = w*weights[argmax], b*weights[argmax]
            else:
                weight, bias = w, b
        else:
            for i, w in enumerate(weights):
                weights_active, bias_active = get_active_weights_bias(bn, feature_dim_list[i], feature_dim_list)
                weight, bias = weight+w*weights_active, bias+w*bias_active
        #print(weight)
        #print(bias)
        return weight, bias
      if bn.num_features == feature_dim or DynamicBatchNorm2dMixture.SET_RUNNING_STATISTICS:
          return bn(x)
      else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            weight, bias = get_params_mixture(bn, weights, feature_dim_list, use_argmax, use_we_v2)
            if use_argmax:
                argmax = np.array([w.item() for w in weights]).argmax()
                out_features = feature_dim_list[argmax]
                if use_we_v2:
                    return F.batch_norm(
                        x[:, :out_features, :, :],
                        bn.running_mean[:out_features],
                        bn.running_var[:out_features],
                        weight,
                        bias,
                        bn.training or not bn.track_running_stats,
                        exponential_average_factor,
                        bn.eps,
                    )
                else:
                    return F.batch_norm(
                    x[:, :out_features, :, :],
                    bn.running_mean[:out_features],
                    bn.running_var[:out_features],
                    weight,
                    bias,
                    bn.training or not bn.track_running_stats,
                    exponential_average_factor,
                    bn.eps,
                )*weights[argmax]
            return F.batch_norm(
                x,
                bn.running_mean,
                bn.running_var,
                weight,
                bias,
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x, weights, use_argmax=False):
        #print("DynamicBatchNorm2dMixture using we_v2: ", self.use_we_v2)
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            out_features = self.feature_dim_list[argmax]
        else:
            out_features = max(self.feature_dim_list)
        y = self.bn_forward(x, self.bn, self.feature_dim_list, feature_dim=out_features, weights=weights, use_argmax=use_argmax, use_we_v2=self.use_we_v2)
        return y


class DynamicGroupNorm(nn.GroupNorm):
    def __init__(
        self, num_groups, num_channels, eps=1e-5, affine=True, channel_per_group=None
    ):
        super(DynamicGroupNorm, self).__init__(
            num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group

    def forward(self, x):
        n_channels = x.size(1)
        n_groups = n_channels // self.channel_per_group
        return F.group_norm(
            x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps
        )

    @property
    def bn(self):
        return self


class DynamicGroupNormMixture(nn.GroupNorm):
    def __init__(
        self, num_groups, channels_list, eps=1e-5, affine=True, channel_per_group=None, use_we_v2=False
    ):
        super(DynamicGroupNormMixture, self).__init__(
            num_groups, channels_list, eps, affine)
        self.channel_per_group = channel_per_group
        self.channels_list = channels_list
        self.max_channels = max(channels_list)
        self.use_we_v2 = use_we_v2

    def get_active_weight(self, out_features):
        return F.pad(self.weight[:out_features], (0, self.max_channels-out_features), "constant", 0)

    def get_active_bias(self, out_features):
        if self.bias is None:
            return self.bias
        else:
            return F.pad(self.bias[:out_features], (0, self.max_channels-out_features), "constant", 0)

    def get_weight_bias_mixture(self, weights, use_argmax=False):
        conv_weight = 0
        conv_bias = 0
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            n_channels = self.channels_list[argmax]
            if self.use_we_v2:
               conv_weight = self.get_active_weight(n_channels)*weights[argmax]
               conv_bias = self.get_active_bias(n_channels)*weights[argmax]
            else:
                conv_weight = self.get_active_weight(n_channels)
                conv_bias = self.get_active_bias(n_channels)
        else:
            for i, w in enumerate(weights):
                n_channels = self.channels_list[i]
                conv_weight = conv_weight + w * \
                    self.get_active_weight(n_channels)
                conv_bias = conv_bias + w * self.get_active_bias(n_channels)

        return conv_weight, conv_bias

    def forward(self, x, weights, use_argmax=False):
        #print("DynamicGroupNormMixture using we_v2: ", self.use_we_v2)
        n_groups = self.max_channels // self.channel_per_group
        weight, bias = self.get_weight_bias_mixture(
            weights, use_argmax=use_argmax)
        argmax = np.array([w.item() for w in weights]).argmax()
        if self.use_we_v2:
            return F.group_norm(
            x, n_groups, weight.to(x.device), bias.to(x.device), self.eps
        )
        else:
            return F.group_norm(
            x, n_groups, weight, bias, self.eps
        )*weights[argmax]

    @property
    def bn(self):
        return self


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.reduce.weight[:num_mid, :in_channel, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.reduce.weight[:num_mid, :, :, :], groups, dim=1
            )
            return torch.cat(
                [sub_filter[:, :sub_in_channels, :, :]
                    for sub_filter in sub_filters],
                dim=1,
            )

    def get_active_reduce_bias(self, num_mid):
        return (
            self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None
        )

    def get_active_expand_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.weight[:in_channel, :num_mid, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.expand.weight[:, :num_mid, :, :], groups, dim=0
            )
            return torch.cat(
                [sub_filter[:sub_in_channels, :, :, :]
                    for sub_filter in sub_filters],
                dim=0,
            )

    def get_active_expand_bias(self, in_channel, groups=None):
        if groups is None or groups == 1:
            return (
                self.fc.expand.bias[:in_channel]
                if self.fc.expand.bias is not None
                else None
            )
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            return torch.cat(
                [sub_bias[:sub_in_channels] for sub_bias in sub_bias_list], dim=0
            )

    def forward(self, x, groups=None):
        in_channel = x.size(1)
        num_mid = make_divisible(
            in_channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE
        )

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel, groups=groups)
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicSEMixture(SEModule):
    def __init__(self, in_channels_list, use_we_v2=False):
        super(DynamicSEMixture, self).__init__(max(in_channels_list))
        self.in_channels_list = in_channels_list
        self.use_we_v2 = use_we_v2
        self.num_mid = [make_divisible(
            c // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE) for c in in_channels_list]
        self.max_num_mid = max(self.num_mid)
        #print(self.num_mid)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None, use_argmax=False):
        if groups is None or groups == 1:
            out = self.fc.reduce.weight[:num_mid, :in_channel, :, :]
            if use_argmax:
                return out
            return F.pad(out, (0,0,0,0,0,max(self.in_channels_list)-in_channel, 0, self.max_num_mid-num_mid), "constant", 0)
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.reduce.weight[:num_mid, :, :, :], groups, dim=1
            )
            out = torch.cat(
                [sub_filter[:, :sub_in_channels, :, :]
                    for sub_filter in sub_filters],
                dim=1,
            )
            if use_argmax:
                return out
            return F.pad(out, (0,0,0,0,0,max(self.in_channels_list)-in_channel, 0, self.max_num_mid-num_mid), "constant", 0)

    def get_active_reduce_bias(self, num_mid, use_argmax=False):
        out = self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None
        if use_argmax:
            return out
        return (F.pad(out, (0, self.max_num_mid-num_mid), "constant", 0))

    def get_active_expand_weight(self, num_mid, in_channel, groups=None, use_argmax=False):
        if groups is None or groups == 1:
            out = self.fc.expand.weight[:in_channel, :num_mid, :, :]
            if use_argmax:
                return out
            return F.pad(out, (0,0,0,0,0,self.max_num_mid-num_mid,0,max(self.in_channels_list)-in_channel), "constant", 0)
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.expand.weight[:, :num_mid, :, :], groups, dim=0
            )
            out = torch.cat(
                [sub_filter[:sub_in_channels, :, :, :]
                    for sub_filter in sub_filters],
                dim=0,
            )
            if use_argmax:
                return out
            return F.pad(out, (0,0,0,0,0,self.max_num_mid-num_mid,0,max(self.in_channels_list)-in_channel), "constant", 0)

    def get_active_expand_bias(self, in_channel, groups=None, use_argmax=False):
        if groups is None or groups == 1:
            out = self.fc.expand.bias[:in_channel] if self.fc.expand.bias is not None else None
            if use_argmax:
                return out
            return (F.pad(out, (0,max(self.in_channels_list)-in_channel), "constant", 0))

        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            out = torch.cat(
                [sub_bias[:sub_in_channels] for sub_bias in sub_bias_list], dim=0
            )
            if use_argmax:
                return out
            return F.pad(out, (0, max(self.in_channels_list)-in_channel), "constant", 0)

    def get_param_mixture(self, weights, use_argmax=False, groups=None):
        reduce_filter = 0
        reduce_bias = 0
        expand_filter = 0
        expand_bias = 0
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            in_channel = self.in_channels_list[argmax]
            num_mid = make_divisible(
                in_channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE)
            if self.use_we_v2:
             reduce_filter = self.get_active_reduce_weight(
                num_mid, in_channel, groups=groups, use_argmax=use_argmax).contiguous() * weights[argmax]
             reduce_bias = self.get_active_reduce_bias(num_mid, use_argmax=use_argmax) * weights[argmax]
             expand_filter = self.get_active_expand_weight(
                num_mid,in_channel, groups=groups,use_argmax=use_argmax).contiguous() * weights[argmax]
             expand_bias = self.get_active_expand_bias(
                in_channel, groups=groups,use_argmax=use_argmax) * weights[argmax]
            else:
                reduce_filter = self.get_active_reduce_weight(
                    num_mid, in_channel, groups=groups, use_argmax=use_argmax).contiguous()
                reduce_bias = self.get_active_reduce_bias(num_mid, use_argmax=use_argmax)
                expand_filter = self.get_active_expand_weight(
                    num_mid,in_channel, groups=groups,use_argmax=use_argmax).contiguous()
                expand_bias = self.get_active_expand_bias(
                    in_channel, groups=groups,use_argmax=use_argmax)
        else:
            for i, w in enumerate(weights):
                in_channel = self.in_channels_list[i]
                num_mid = make_divisible(in_channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE)
                reduce_filter = reduce_filter + w * self.get_active_reduce_weight(num_mid, in_channel, groups=groups).contiguous()
                reduce_bias = reduce_bias + w * self.get_active_reduce_bias(num_mid)
                expand_filter = expand_filter + w * self.get_active_expand_weight(num_mid, in_channel, groups=groups).contiguous()
                expand_bias = expand_bias + w * self.get_active_expand_bias(in_channel, groups=groups)
        return reduce_filter, reduce_bias, expand_filter, expand_bias

    def forward(self, x, weights, use_argmax=False, groups=None):
        #print("DynamicSEMixture using use_we_v2 = ", self.use_we_v2)
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_filter, reduce_bias, expand_filter, expand_bias = self.get_param_mixture(weights, use_argmax=use_argmax, groups=groups
        )
        #print(reduce_filter.shape)
        #print(reduce_bias.shape)
        #print(reduce_bias)
        if use_argmax:
            selected = np.array([w.item() for w in weights]).argmax()
            in_channel = self.in_channels_list[selected]
            #print(reduce_filter)
            if self.use_we_v2:
                y = F.conv2d(y[:, :in_channel, :, :], reduce_filter, reduce_bias, 1, 0, 1, 1)
            else:
                y = F.conv2d(y[:, :in_channel, :, :], reduce_filter, reduce_bias, 1, 0, 1, 1)*weights[selected]
        else:
            y = F.conv2d(y, reduce_filter.to(x.device), reduce_bias.to(x.device), 1, 0, 1, 1)
        # relu
        #print(expand_filter)
        #print(expand_bias)
        #print(expand_bias.shape)
        #print(expand_filter.shape)
        y = self.fc.relu(y)
        # expand
        if use_argmax:
            selected = np.array([w.item() for w in weights]).argmax()
            #print(expand_bias)
            #print(expand_filter)
            if self.use_we_v2:
                y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
            else:
                y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)*weights[selected]
        else:
            y = F.conv2d(y, expand_filter.to(x.device), expand_bias.to(x.device), 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)
        if use_argmax:
            x = x[:, :in_channel, :, :]
            return x * y
        return x * y


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features,
                                self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y


class DynamicLinearMixture(nn.Module):
    def __init__(self, max_in_features, max_out_features, in_features_list, out_features_list, bias=True, use_we_v2=False):
        super(DynamicLinearMixture, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.use_we_v2 = use_we_v2

        self.linear = nn.Linear(self.max_in_features,
                                self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features, use_argmax=False):
        if use_argmax:
            return self.linear.weight[:out_features, :in_features]
        return F.pad(self.linear.weight[:out_features, :in_features], (0, self.max_in_features-in_features, 0, self.max_out_features-out_features), "constant", 0)

    def get_active_bias(self, out_features, use_argmax=False):
        if self.linear.bias is None:
            return self.bias
        else:
            if use_argmax:
                return self.linear.bias[:out_features]
            return F.pad(self.linear.bias[:out_features], (0, self.max_out_features-out_features), "constant", 0)

    def get_weight_bias_mixture(self, weights, use_argmax=False):
        conv_weight = 0
        conv_bias = 0
        channels_cross = list(itertools.product(
            self.in_features_list, self.out_features_list))
        if use_argmax == True:
            argmax = np.array([w.item() for w in weights]).argmax()
            in_channels, out_channels = channels_cross[argmax]
            if self.use_we_v2:
               conv_weight = self.get_active_weight(out_channels, in_channels, use_argmax=use_argmax) * weights[argmax]
               conv_bias = self.get_active_bias(out_channels, use_argmax=use_argmax) * weights[argmax]
            else:
                conv_weight = self.get_active_weight(out_channels, in_channels, use_argmax=use_argmax)
                conv_bias = self.get_active_bias(out_channels, use_argmax=use_argmax)
        else:
            for i, w in enumerate(weights):
                in_channels, out_channels = channels_cross[i]
                conv_weight = conv_weight + w * \
                    self.get_active_weight(out_channels, in_channels)
                conv_bias = conv_bias + w * self.get_active_bias(out_channels)
        return conv_weight.contiguous(), conv_bias

    def forward(self, x, weights, use_argmax=False):
        print("Linear using use_we_v2: ", self.use_we_v2)
        weights_curr, bias_curr = self.get_weight_bias_mixture(
            weights, use_argmax=use_argmax)
        channels_cross = list(itertools.product(
            self.in_features_list, self.out_features_list))
        if use_argmax:
            selected = np.array([w.item() for w in weights]).argmax()
            in_channels, out_channels = channels_cross[selected]
            x = x[:, :in_channels]
            #print(weights_curr)
            #print(bias_curr)
            if self.use_we_v2:
               y = F.linear(x, weights_curr, bias_curr)
            else:
                y = F.linear(x, weights_curr, bias_curr)*weights[selected]
        else:
            y = F.linear(x, weights, bias_curr)
        return y

# test DynamicOutConv2dMixture
'''import torch
dynamicOutCov2dMixture = DynamicOutConv2dMixture(2, [2,4,6], kernel_size=1, stride=1, dilation=1, use_we_v2=False)
dynamicOutCov2dMixture.conv.weight.data = torch.ones_like(dynamicOutCov2dMixture.conv.weight.data)
input = torch.ones([1,2,2,2])
weights = torch.tensor([0.6,0.3,0.1])
output = dynamicOutCov2dMixture(input,weights,use_argmax=True)
print(output.shape)
print(output)
# Testing steps
# 1. Forward prop and check that kernel mixture is correct
# 2. Check if every arch param has gradient
# 3. COmpare individual componets with base OFA model 

dynamicbatchnorm = DynamicBatchNorm2dMixture([2,3,4],use_we_v2=False)
dynamicbatchnorm.bn.weight.data = torch.ones_like(dynamicbatchnorm.bn.weight.data)
dynamicbatchnorm.bn.bias.data = torch.ones_like(dynamicbatchnorm.bn.bias.data)
input = torch.ones([2,4,2,2])
weights = torch.tensor([0.1,0.7,0.2])
out = dynamicbatchnorm(input,weights,use_argmax=True)
print(out.shape)

dynamic_in_separable = DynamicInSeparableConv2dMixture([2,3,4],[3,5,7],1,use_we_v2=False)
dynamic_in_separable.conv.weight.data = torch.ones_like(dynamic_in_separable.conv.weight.data)
#dynamic_in_separable.conv.bias.data  = torch.ones_like(dynamic_in_separable.conv.bias.data)
input = torch.ones([2,4,8,8])
weights = torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1])
out = dynamic_in_separable(input,weights,use_argmax=True)
print(out.shape)
in_channels_list = [4,6,8]
se_mixture = DynamicSEMixture(in_channels_list, use_we_v2=False)
se_mixture.fc.reduce.weight.data = torch.ones_like(se_mixture.fc.reduce.weight)
se_mixture.fc.reduce.bias.data = torch.ones(se_mixture.fc.reduce.bias.shape)
se_mixture.fc.expand.weight.data = torch.ones_like(se_mixture.fc.expand.weight)
se_mixture.fc.expand.bias.data = torch.ones_like(se_mixture.fc.expand.bias)
input = torch.ones([2,8,4,4])
weights = torch.tensor([0.6,0.2,0.2])
out = se_mixture(input,weights,use_argmax=True)
print(out.shape)'''
# test dynamiclinear
'''dynamiclinear = DynamicLinearMixture(4,6,[2,3,4],[3,5,7],use_we_v2=False)
dynamiclinear.linear.weight.data = torch.ones_like(dynamiclinear.linear.weight.data)
dynamiclinear.linear.bias.data = torch.ones_like(dynamiclinear.linear.bias.data)
input = torch.ones([2,4])
weights = torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1])
out = dynamiclinear(input,weights,use_argmax=True)'''
