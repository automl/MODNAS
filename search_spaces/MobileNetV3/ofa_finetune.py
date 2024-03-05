# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
from torchsummary import summary
import copy
import random

from search_spaces.MobileNetV3.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
)
from search_spaces.MobileNetV3.modules.dynamic_layers import (
    DynamicMBConvLayer,
)
from predictors.help.net import  MetaLearner
from search_spaces.MobileNetV3.mobilenetv3 import MobileNetV3
from search_spaces.MobileNetV3.utils import make_divisible, val2list, MyNetwork
from optimizers.optim_factory import get_sampler
import torch.nn as nn
from hypernetworks.models.hpn_ofa import MetaHyperNetwork, HyperNetwork
from predictors.help.loader import Data
from predictors.help.net import  MetaLearner
from predictors.help.loader import Data
from predictors.help.utils import get_minmax_latency_index
from search_spaces.MobileNetV3.utils.pytorch_utils import cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
__all__ = ["OFAMobileNetV3"]


class OFAMobileNetV3(MobileNetV3):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        base_stage_width=None,
        width_mult=1.2,
        ks_list=3,
        expand_ratio_list=6,
        depth_list=4,
        image_size_list = [224, 192, 160, 128, 96],
    ):

        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.image_size_list = image_size_list

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.image_size_list.sort()
        self.sampler = get_sampler("reinmax")
        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        self.help_loader = Data(mode="nas",data_path="datasets/help/data/ofa",search_space="ofa", meta_valid_devices=[], meta_test_devices= [], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )
        final_expand_width = make_divisible(
            base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        #print("final_expand_width", final_expand_width)
        last_channel = make_divisible(
            base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        #print("last_channel", last_channel

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(
                base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
            )
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, act_func="h_swish"
        )
        first_block_conv = MBConvLayer(
            in_channels=input_channel,
            out_channels=first_block_dim,
            kernel_size=3,
            stride=stride_stages[0],
            expand_ratio=1,
            act_func=act_stages[0],
            use_se=se_stages[0],
        )
        first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(first_block_dim, first_block_dim)
            if input_channel == first_block_dim
            else None,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = first_block_dim

        for width, n_block, s, act_func, use_se in zip(
            width_list[2:],
            n_block_list[1:],
            stride_stages[1:],
            act_stages[1:],
            se_stages[1:],
        ):
            self.block_group_info.append(
                [_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        # final expand layer, feature mix layer & classifier
        final_expand_layer = ConvLayer(
            feature_dim, final_expand_width, kernel_size=1, act_func="h_swish"
        )
        feature_mix_layer = ConvLayer(
            final_expand_width,
            last_channel,
            kernel_size=1,
            bias=False,
            use_bn=False,
            act_func="h_swish",
        )

        classifier = LinearLayer(
            last_channel, n_classes, dropout_rate=dropout_rate)
        

        super(OFAMobileNetV3, self).__init__(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        self.predictor =  MetaLearner("ofa",True,10,100)
        for param in self.predictor.parameters():
            param.requires_grad = False
        # runtime_depth
        self.runtime_depth = [len(block_idx)
                              for block_idx in self.block_group_info]
        self.ce_trajectory = []
        self.latency_trajectory = []

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return "OFAMobileNetV3"
    
    def convert_to_resoltuion(self ,x,  r):
        out =  nn.functional.interpolate(x, size=(r,r), mode='bilinear', align_corners=True)
        return out

    def get_gt_stats_latency(self, device):
        import pickle
        with open("predictor_data_utils/ofa/stats_ofa.pkl","rb") as f:
            stats = pickle.load(f)
        #print(stats[device]["max"], stats[device]["min"])
        return stats[device]["max"], stats[device]["min"]
    
    
    def preprocess_for_predictor(self, arch_params):
        self.predictor.load_state_dict(torch.load("predictor_data_utils/ofa/ofa_predictor_modified.pt"), strict=False)
        kernel_size_weights, expand_ratio_weights, depth_weights, resolution_weights = arch_params["ks"], arch_params["e"], arch_params["d"], arch_params["r"]
        depth_selected = []
        #print(depth_weights)
        kernel_size_zeros = torch.zeros_like(kernel_size_weights).to(kernel_size_weights.device)
        expand_ratio_zeros = torch.zeros_like(expand_ratio_weights).to(kernel_size_weights.device)
        for i in range(len(self.block_group_info)):
            argmax_depth = torch.argmax(depth_weights[i])
            depth_selected.append(self.depth_list[argmax_depth])
        start = 0
        end = max(self.depth_list)
        for i,d in enumerate(depth_selected):
            for j in range(start,start+d):
                expand_ratio_zeros[i,j,:] = expand_ratio_weights[i,j,:]
                kernel_size_zeros[i,j,:] = kernel_size_weights[i,j,:]
            for j in range(start+d, end):
                expand_ratio_zeros[i,j,:] = 0
                kernel_size_zeros[i,j,:] = 0
        #resolution_weights = torch.zeros_like(arch_params["r"])
        # set max
        #resolution_weights[-1] = 1
        out = torch.cat([kernel_size_zeros.reshape(-1),expand_ratio_zeros.reshape(-1), resolution_weights.reshape(-1), depth_weights.reshape(-1)]).unsqueeze(0)
        return out
    
    def reset_ce_stats(self):
        self.ce_trajectory = []

    def get_max_arch_one_hot(self, vector):
        result = torch.zeros(vector.shape).to(vector.device)
        # set last elemnt to 1
        if len(vector.shape) == 1:
            result[-1] = 1
        result[...,-1] = 1
        return result
    
    def forward(self, x, labels, hw_embed, arch_params=None, device = "", set_running_statistics=False):
        # first conv
        #print(arch_params)
        if set_running_statistics:
            arch_params_sampled = {}
            for k in arch_params.keys():
                arch_params_sampled[k] = self.get_max_arch_one_hot(arch_params[k])
        elif self.training:
            arch_params_sampled = {}
            for key in arch_params:
                arch_params_sampled[key] = self.sampler.sample(arch_params[key])
        else:
            arch_params_sampled = {}
            #print(arch_params)
            for key in arch_params:
                arch_params_sampled[key] = torch.zeros_like(arch_params[key])
                if len(arch_params[key].shape) == 1:
                    argmax = torch.argmax(arch_params[key], dim=-1).item()
                    arch_params_sampled[key][argmax] = 1
                elif len(arch_params[key].shape) == 2:
                    for i in range(arch_params[key].shape[0]):
                        argmax = torch.argmax(arch_params[key][i], dim=-1).item()
                        arch_params_sampled[key][i, argmax] = 1      
                elif len(arch_params[key].shape) == 3:
                    for i in range(arch_params[key].shape[0]):
                        for j in range(arch_params[key].shape[1]):
                            argmax = torch.argmax(arch_params[key][i,j], dim=-1).item()
                            arch_params_sampled[key][i,j, argmax] = 1
        #print(arch_params_sampled)
        #arch_params = arch_params_sampled
        predictor_input = self.preprocess_for_predictor(arch_params_sampled)
        latency = self.predictor(predictor_input,hw_embed)
        self.set_active_subnet(arch_params_sampled)
        resolution = arch_params_sampled["r"]
        resolution_param_argmax = torch.argmax(resolution, dim=-1)
        
        if self.image_size_list[resolution_param_argmax] != 224:
           x = self.convert_to_resoltuion(x, self.image_size_list[resolution_param_argmax])
        resolution = arch_params_sampled["r"]
        x = x * resolution[resolution_param_argmax]
        depth_param = arch_params_sampled["d"]
        depth_param_argmax = torch.argmax(depth_param, dim=-1)
        #depths_selected = [self.depth_list[i] for i in depth_param_argmax]
        depth_max_param = [depth_param[i,depth_param_argmax[i]] for i in range(len(self.block_group_info))]
        kernel_param = arch_params_sampled["ks"].reshape(5*4,3)
        kernel_param_argmax = torch.argmax(kernel_param, dim=-1)
        kernel_max_param = []
        expand_param = arch_params_sampled["e"].reshape(5*4,3)
        expand_param_argmax = torch.argmax(expand_param, dim=-1)
        expand_max_param = []
        for i in range(len(self.blocks) - 1):
            kernel_max_param.append(kernel_param[i, kernel_param_argmax[i]])
            expand_max_param.append(expand_param[i, expand_param_argmax[i]])

        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)
        # blocks
        count = 0
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x, kernel_max_param[count], expand_max_param[count])
                count += 1
            for no_idx in range(len(block_idx) - depth):
                count += 1
            x = x * depth_max_param[stage_id]
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(
            2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        ce_loss = cross_entropy_with_label_smoothing(
                    x, labels, 0.1
                )
        if len(self.ce_trajectory) >1:
            ce_normalized = ce_loss
            max_lat, min_lat = self.get_gt_stats_latency(device)
            max_lat = max_lat.item()
            min_lat = min_lat.item()
            latency_normalized = (latency-min_lat)/(max_lat-min_lat)
            latency_normalized = latency_normalized*(max(self.ce_trajectory)-min(self.ce_trajectory)) + min(self.ce_trajectory)
        else:
            ce_normalized = ce_loss
            latency_normalized = latency    
        if self.training and set_running_statistics==False:
           self.ce_trajectory.append(ce_loss.item())
        if not self.training:
            latency_normalized = latency
        if len(self.ce_trajectory) >200:
            self.reset_ce_stats()
        return x, ce_normalized, latency_normalized

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"

        _str += self.final_expand_layer.module_str + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": OFAMobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "final_expand_layer": self.final_expand_layer.config,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if ".mobile_inverted_conv." in key:
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif ".bn.bn." in new_key:
                new_key = new_key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in new_key:
                new_key = new_key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in new_key:
                new_key = new_key.replace(".linear.linear.", ".linear.")
            ##############################################################################
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAMobileNetV3, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list)
        )

    def set_best_net(self):
        self.set_active_subnet(
            ks=[7,5,5,7,5,5,7,7,5,7,7,7,5,7,7,5,5,7,7,5], e=[6, 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6], d=[4,4,4,4,4]
        , preserve_weight=True)

    def set_active_subnet(self, arch_param, **kwargs):
        resoluton_argmax = torch.argmax(arch_param["r"], dim=-1)
        self.resolution = self.image_size_list[resoluton_argmax]
        d_param = arch_param["d"]
        d_param_argmax = torch.argmax(d_param, dim=-1)
        depth = [self.depth_list[i] for i in d_param_argmax]
        ks_param = arch_param["ks"]
        ks_param_argmax = torch.argmax(ks_param.reshape(5*4,3), dim=-1)
        ks = []
        expand_ratio = []
        e_param = arch_param["e"]
        e_param_argmax = torch.argmax(e_param.reshape(5*4,3), dim=-1)
        #print(len(depth))
        #print(len(self.blocks) - 1)
        for i in range(len(self.blocks) - 1):
                ks.append(self.ks_list[ks_param_argmax[i]])
                expand_ratio.append(self.expand_ratio_list[e_param_argmax[i]])

                

        #ks = val2list(ks, len(self.blocks) - 1)
        #expand_ratio = val2list(e, len(self.blocks) - 1)
        #depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                # print(block.conv)
                block.conv.active_kernel_size = k
            if e is not None:
                # print(block.conv)
                block.conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type="depth"):
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None

    def sample_active_subnet(self):
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [
                ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [
                expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]

        final_expand_layer = copy.deepcopy(self.final_expand_layer)
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    ResidualBlock(
                        self.blocks[idx].conv.get_active_subnet(
                            input_channel, preserve_weight
                        ),
                        copy.deepcopy(self.blocks[idx].shortcut),
                    )
                )
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        final_expand_config = self.final_expand_layer.config
        feature_mix_layer_config = self.feature_mix_layer.config
        classifier_config = self.classifier.config

        block_config_list = [first_block_config]
        input_channel = first_block_config["conv"]["out_channels"]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    {
                        "name": ResidualBlock.__name__,
                        "conv": self.blocks[idx].conv.get_active_subnet_config(
                            input_channel
                        ),
                        "shortcut": self.blocks[idx].shortcut.config
                        if self.blocks[idx].shortcut is not None
                        else None,
                    }
                )
                input_channel = self.blocks[idx].conv.active_out_channel
            block_config_list += stage_blocks

        return {
            "name": MobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": first_conv_config,
            "blocks": block_config_list,
            "final_expand_layer": final_expand_config,
            "feature_mix_layer": feature_mix_layer_config,
            "classifier": classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)
