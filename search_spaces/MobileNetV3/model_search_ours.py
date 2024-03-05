# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
import copy
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from predictors.help.utils import arch_encoding_ofa
import pickle
from search_spaces.MobileNetV3.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
    ResidualBlockMix
)
from search_spaces.MobileNetV3.modules.dynamic_layers import (
    DynamicMBConvLayer, DynamicMBConvLayerMixture
)
import torch.distributed as dist
from torch.autograd import Variable
from search_spaces.MobileNetV3.mobilenetv3 import MobileNetV3
from search_spaces.MobileNetV3.utils import make_divisible, val2list, MyNetwork
from optimizers.optim_factory import get_mixop, get_sampler
from predictors.help.net import  MetaLearner
from predictors.help.loader import Data
from predictors.help.utils import get_minmax_latency_index
from search_spaces.MobileNetV3.utils.pytorch_utils import cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
__all__ = ["OFAMobileNetV3Mixture"]


class OFAMobileNetV3Mixture(MobileNetV3):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        base_stage_width=None,
        width_mult=1.2,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[1, 2, 3],
        optimizer = "darts_v1",
        use_we_v2 = False,
        latency_norm_scheme = "batch_stats",
    ):
        self.resolution_list = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224]
        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.use_we_v2 = use_we_v2
        self.help_loader = Data(mode="nas",data_path="datasets/help/ofa",search_space="ofa", meta_valid_devices=[], meta_test_devices= [], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        self.mixop = get_mixop(optimizer)
        self.sampler = get_sampler(optimizer)

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = make_divisible(
            base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        last_channel = make_divisible(
            base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )

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
                mobile_inverted_conv = DynamicMBConvLayerMixture(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    mixop=self.mixop,
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                    use_we_v2=use_we_v2
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlockMix(mobile_inverted_conv, shortcut))
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

        super(OFAMobileNetV3Mixture, self).__init__(
            first_conv, blocks, final_expand_layer, feature_mix_layer, classifier
        )
        #self._init_arch_parameters()
        # set bn param
        self.ce_loss = nn.CrossEntropyLoss()
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx)
                              for block_idx in self.block_group_info]
        self.predictor =  MetaLearner("ofa",True,10,100)
        self.predictor.load_state_dict(torch.load("predictor_data_utils/ofa/ofa_predictor_modified.pt", map_location=torch.device('cpu')))
        for param in self.predictor.parameters():
            param.requires_grad = False
        #self.hpn = MetaHyperNetwork(len(self.block_group_info),max_depth=max(self.depth_list), kernel_list_len=len(self.ks_list), expand_ratio_list_len=len(self.expand_ratio_list), depth_list_len=len(self.depth_list), resolution_len=len(self.resolution_list))
        self.model_name = "mobilenetv3_modnas"
        self.ce_trajectory = []
        self.latency_trajectory = []
        self.latency_norm_scheme = latency_norm_scheme

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return "OFAMobileNetV3ixture"

    def get_depth_mixture(self, all_features, depth_weights):
        out = 0
        for i, w in enumerate(depth_weights):
            out = out + all_features[i] * w
        return out
    
    def preprocess_for_predictor(self, arch_params):
        self.predictor.load_state_dict(torch.load("predictor_data_utils/ofa/ofa_predictor_modified.pt"))
        kernel_size_weights, expand_ratio_weights, depth_weights, resolution_weights = arch_params[0], arch_params[1], arch_params[2], arch_params[3]
        depth_selected = []
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
        
        out = torch.cat([kernel_size_zeros.reshape(-1),expand_ratio_zeros.reshape(-1), resolution_weights.reshape(-1), depth_weights.reshape(-1)]).unsqueeze(0)
        return out

    def _init_arch_parameters(self):
        arch_kernel_size_weights = torch.nn.Parameter(1e-3 * torch.randn(
            [len(self.block_group_info), max(self.depth_list), len(self.ks_list)]),
            requires_grad=True
        )
        arch_expand_ratio_weights = torch.nn.Parameter(1e-3 * torch.randn(
            [
                len(self.block_group_info),
                max(self.depth_list),
                len(self.expand_ratio_list)
            ]),
            requires_grad=True
        )
        arch_depth_weights = torch.nn.Parameter(1e-3 * torch.randn(
            [len(self.block_group_info), len(self.depth_list)]),
            requires_grad=True
        )

        arch_resolutions = torch.nn.Parameter(1e-3 * torch.randn([len(self.resolution_list)]), requires_grad=True)
        return [arch_kernel_size_weights,arch_expand_ratio_weights,arch_depth_weights,arch_resolutions]
        #print(len(self.block_group_info), max(self.depth_list), len(self.ks_list), len(self.expand_ratio_list), len(self.depth_list))
        #self.register_buffer('arch_kernel_size_weights', arch_kernel_size_weights)
        #self.register_buffer('arch_expand_ratio_weights', arch_expand_ratio_weights)
        #self.register_buffer('arch_depth_weights', arch_depth_weights)
    
    def model_weights(self):
        model_weights = []
        for name, param in self.named_parameters():
            if "predictor" not in name and "hpn" not in name:
                model_weights.append(param)
        return model_weights
    
    def get_model_params(self, key, include=False):
        param_list = []
        if include:
            for name, param in self.named_parameters():
                if (key in name) and ('arch' not in name):
                    param_list.append(param)
        else:
            for name, param in self.named_parameters():
                if (key not in name) and ('arch' not in name):
                    param_list.append(param)
        return param_list

    def get_arch_represenation(self, arch_params):
        def get_argmaxes(weights):
            # count the number of dimensions in weights
            num_dims = len(weights.shape)

            if num_dims == 1:
                return torch.argmax(weights, dim=0, keepdim=False).detach().cpu().numpy()

            if num_dims == 2:
                return torch.argmax(weights, dim=1, keepdim=False).detach().cpu().numpy()
            elif num_dims == 3:
                arg_maxs = []

                for block_weights in weights:
                    arg_maxs.append(torch.argmax(block_weights, dim=1, keepdim=False).detach().cpu().tolist())

                return np.array(arg_maxs)

        def convert_to_genotype(alphas_argmax, operations):
            return np.array(
                [operations[x] for x in alphas_argmax.reshape(-1).tolist()]
            ).reshape(alphas_argmax.shape).tolist()

        def convert_genotype_to_dict(genotype):
            blocks = {}
            #print(genotype)

            g = np.array(genotype).squeeze()
            num_dims = len(g.shape)
            if num_dims == 1 or num_dims==0:
                return  genotype
            for i, block in enumerate(g):
                blocks[f"block_{i}"] = block.tolist()

            return blocks
        
        kernel_size_weights, expand_ratio_weights, depth_weights, resolution_weights = arch_params[0], arch_params[1], arch_params[2], arch_params[3]
        
        arch_configs = [
            (kernel_size_weights, self.ks_list, "kernel_size"),
            (expand_ratio_weights, self.expand_ratio_list, "expand_ratio"),
            (depth_weights, self.depth_list, "depth"),
            (resolution_weights, self.resolution_list, "resolution")
        ]

        genotype_draft = {}
     
        for arch_params, choices, op_name in arch_configs:
            arg_maxs = get_argmaxes(arch_params)
            genotype = convert_to_genotype(arg_maxs, choices)
            genotype_draft[op_name] = genotype
        representation = {}

        for k, v in genotype_draft.items():
            representation[k] = convert_genotype_to_dict(v)

        return representation
    
    def convert_to_one_hot(self,vector):
        result = torch.zeros(vector.shape).to(vector.device)
        # set argmax along last axis to 1
        #print(vector.argmax(dim=-1, keepdim=True))
        result.scatter_(-1, vector.argmax(dim=-1, keepdim=True), 1)
        return result
    
    def get_max_arch_one_hot(self, vector):
        result = torch.zeros(vector.shape).to(vector.device)
        # set last elemnt to 1
        if len(vector.shape) == 1:
            result[-1] = 1
        result[...,-1] = 1
        return result

    def reset_ce_stats(self):
        self.ce_trajectory = []

    def get_gt_stats_latency_2(self, device):
        with open("predictor_data_utils/ofa/stats_ofa.pkl","rb") as f:
            stats = pickle.load(f)
        #print(stats[device]["max"], stats[device]["min"])
        return stats[device]["max"], stats[device]["min"]


    def get_gt_stats_latency(self, device):
        max_lat = max(self.help_loader.latency[device][[self.help_loader.max_lat_idx,self.help_loader.min_lat_idx]+self.help_loader.hw_emb_idx])
        min_lat = min(self.help_loader.latency[device][[self.help_loader.max_lat_idx,self.help_loader.min_lat_idx]+self.help_loader.hw_emb_idx])
        #print(max_lat,min_lat)
        #self.get_gt_stats_latency_2(device)
        return max_lat, min_lat



    def forward(self, x, labels, hw_embed, arch_params=None, device = "", set_running_statistics=False):
        # first conv
        #arch_params = self.hpn(scalarization, hw_embed)
        
        if set_running_statistics:
            weights = [ self.get_max_arch_one_hot(arch_params[i]) for i in range(len(arch_params))]
            #print(weights)
        elif self.training is True:
            #print("Here")
            weights = self.sampler.sample_step(arch_params)
        else:
            #print("here")
            weights = [self.convert_to_one_hot(arch_params[i]) for i in range(len(arch_params))]
            #print(weights)

        kernel_size_weights, expand_ratio_weights, depth_weights, resolution_weights = weights[0], weights[1], weights[2], torch.squeeze(weights[3])
        #dist.broadcast(kernel_size_weights,0)
        #dist.broadcast(expand_ratio_weights,0)
        #dist.broadcast(depth_weights,0)
        #dist.broadcast(resolution_weights,0)
        #print(kernel_size_weights)
        #print(expand_ratio_weights)
        #print(depth_weights)
        #print(resolution_weights)
        # resolution argmax 
        #print(kernel_size_weights.shape)
        #print(expand_ratio_weights.shape)
        #print(depth_weights.shape)
        #print(resolution_weights.shape)
        resolution_argmax = torch.argmax(resolution_weights)
        resolution_w = resolution_weights[resolution_argmax]
        x = x * resolution_w
        #print("Representation", self.get_arch_represenation(weights))
        predictor_input = self.preprocess_for_predictor(weights)
        #print( predictor_input)
        latency = self.predictor(predictor_input,hw_embed)
        #print("Latency predicted", latency)
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)
        # blocks
        for i, block_idx in enumerate(self.block_group_info):
            depths_collector = []
            count = 0
            argmax_depth = torch.argmax(depth_weights[i])
            argmax_weight = depth_weights[i][argmax_depth]
            for idx in list(block_idx):
                x = self.blocks[idx](x, expand_ratio_weights[i,count,:], kernel_size_weights[i,count,:])
                if count+1 == self.depth_list[argmax_depth]:
                    x = x * argmax_weight
                    break
                count = count+1
            #x = self.get_depth_mixture(depths_collector, depth_weights[i])
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(
            2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #soft_labels = F.softmax(x, dim=1)
        ce_loss = cross_entropy_with_label_smoothing(
                    x, labels, 0.1
                )

        if len(self.ce_trajectory) >1:
            ce_normalized = (ce_loss - min(self.ce_trajectory))/(max(self.ce_trajectory)-min(self.ce_trajectory))
            if self.latency_norm_scheme == "batch_stats":
               latency_normalized = (latency - min(self.latency_trajectory))/(max(self.latency_trajectory)-min(self.latency_trajectory))
            else:
                max_lat, min_lat = self.get_gt_stats_latency(device)
                max_lat = max_lat.item()
                min_lat = min_lat.item()
                #print(latency)
                #print(max_lat)
                #print(min_lat)
                latency_normalized = (latency-min_lat)/(max_lat-min_lat)
        else:
            ce_normalized = ce_loss
            latency_normalized = latency
        if len(self.ce_trajectory)>100:
            self.reset_ce_stats()
        #print(self.ce_trajectory)
        #else:
        #max_lat, min_lat = self.get_gt_stats_latency(device)
        #max_lat = max_lat.item()
        #min_lat = min_lat.item()
        #latency_normalized = (latency-min_lat)/(max_lat-min_lat)
        #ce_normalized = ce_loss
        #print("CE loss", ce_loss)
        #latency_normalized = latency
        if self.training and set_running_statistics==False:
           self.ce_trajectory.append(ce_loss.item())
           self.latency_trajectory.append(latency.item())
        return x, ce_normalized, latency_normalized, [kernel_size_weights, expand_ratio_weights, depth_weights, resolution_weights]

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
            "name": OFAMobileNetV3Mixture.__name__,
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
        super(OFAMobileNetV3Mixture, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list)
        )

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
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

'''from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1235'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="drnas").to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(2,3,224,224)).to(rank)
    labels = torch.randn(2, 1000).to(rank)
    if rank==0:
        print(loss_fn(outputs, labels))
    loss_fn(outputs, labels).backward()
    for name, param in ddp_model.named_parameters():
        if param.grad==None:
            print(name, param.grad)
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 8)'''
def sample_arch(ks_list=[3, 5, 7], depth_list=[
                              2,3,4], expand_ratio_list=[3, 4, 6], resolution = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224] ):
    ks = [random.choice(ks_list) for i in range(20)]
    e =  [random.choice(expand_ratio_list) for i in range(20)]
    d = [random.choice(depth_list) for d in range(5)]
    r = random.choice(resolution)
    return {"ks":ks, "e":e, "d":d, "r":r}

if __name__ == "__main__":
 model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              2,3,4], expand_ratio_list=[3, 4, 6], optimizer="reinmax")
 #for k in state_dict:
 #    print(k)
 help_loader =  Data(mode="meta-train",data_path="datasets/ofa",search_space="ofa", meta_valid_devices=[], meta_test_devices= [], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )
 model.sampler.set_taus(0.1,10)
 model.sampler.set_total_epochs(100)
 model.sampler.before_epoch()
 labels = torch.randn(2, 1000)
 scalarization = torch.tensor([[0.8,0.2]])
 optimizer_net = torch.optim.SGD(model.model_weights(), lr=0.001)
 arch_params = model._init_arch_parameters()
 hw_embed, _, _, _, _, _ = help_loader.get_task("titan_rtx_32")
 hw_embed = hw_embed.unsqueeze(0)
 output, loss_ce, loss_latency, arch_params, _ = model(torch.randn(2,3,224,224), labels, hw_embed,arch_params)
 loss = loss_ce+loss_latency
 print(loss)
 print(loss_latency)
 loss.backward()
 #print(model.get_arch_represenation(arch_params))
 archs = torch.load("datasets/help/ofa/ofa_archs.pt")["arch"]
 archs_list = []
 for i in range(len(archs)):
    archs_list.append(arch_encoding_ofa(archs[i]))
 import copy
 for i in range(1000000):
  #print("Arch ", i)

  curr_arch = arch_encoding_ofa(sample_arch())
  current_arch_one_hot = torch.tensor(curr_arch)
  #print(current_arch_one_hot)
  arch_params = [current_arch_one_hot[:5*4*3].reshape(5,4,3),current_arch_one_hot[5*4*3:(2*5*3*4)].reshape(5,4,3),current_arch_one_hot[((2*5*3*4)+25):].reshape(5,3),current_arch_one_hot[2*5*3*4:((2*5*3*4)+25)]]
  kernel_size_weights = arch_params[0]
  expand_weights = arch_params[1]
  #print(kernel_size_weights)
  for k in range(5):
     for l in range(4):
        if torch.sum(kernel_size_weights[k,l,:])==0:
             kernel_size_weights[k,l,1] = 1
             expand_weights[k,l,1] = 1
  arch_params[0] = kernel_size_weights
  arch_params[1] = expand_weights
  model.eval()
  #print(archs[i])
  #print(archs_list[i])
  output, loss_ce, loss_latency, arch_params, pred_inp = model(torch.randn(2,3,224,224), labels, hw_embed, arch_params)
  #print(loss_latency)
  #print("ped inp", pred_inp)
  #print(model.predictor(archs_list[i].unsqueeze(0),hw_embed))
  #print(archs_list[i])
  #print(archs_list[i] == pred_inp)
  assert torch.sum(curr_arch == pred_inp) == len(curr_arch)
  assert loss_latency == model.predictor(curr_arch.unsqueeze(0),hw_embed)
  print(loss_latency)
  #print(help_loader.latency["titan_rtx_32"][i])
  assert loss_latency.item()>0
  print("Arch passed")
 tensor_pred = torch.tensor([[1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
         0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.]])
 print(model.predictor(tensor_pred,hw_embed))
#model.predictor(

#print("Before", model.arch_depth_weights)
#for p in model.arch_parameters():
#    p.grad = torch.ones_like(p)*10
#optimizer.step()
#print("After", model.arch_depth_weights)
#torch.save(model.arch_parameters(), "ofa_mbv3_arch.pth")
#print(torch.load("ofa_mbv3_arch.pth"))
#for n,p in model.named_parameters():
#    print(n)
#for n,p in model.model_weights():
#    print(n)

'''from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1235'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="drnas").to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(2,3,224,224)).to(rank)
    labels = torch.randn(2, 1000).to(rank)
    if rank==0:
        print(loss_fn(outputs, labels))
    loss_fn(outputs, labels).backward()
    for name, param in ddp_model.named_parameters():
        if param.grad==None:
            print(name, param.grad)
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

'''

'''model = OFAMobileNetV3Mixture(ks_list=[3, 5, 7], depth_list=[
                              1, 2, 3], expand_ratio_list=[3, 4, 6], optimizer="reinmax",use_we_v2=
                              False)
print(model.sampler)
model.sampler.set_taus(0.1,10)
model.sampler.set_total_epochs(100)
model.sampler.before_epoch()
#for k in state_dict:
#    print(k)
arch
outputs = model(torch.randn(2,3,224,224), torch.randn(2, 1000), torch.randn(1,10))
optimizer = torch.optim.SGD(model.arch_parameters(), lr=0.001)
loss = torch.sum(outputs)
loss.backward()'''
#print("Before", model.arch_depth_weights)
#for p in model.arch_parameters():
#    p.grad = torch.ones_like(p)*10
#optimizer.step()
#print("After", model.arch_depth_weights)
#torch.save(model.arch_parameters(), "ofa_mbv3_arch.pth")
#print(torch.load("ofa_mbv3_arch.pth"))'''
#for n,p in model.named_parameters():
#    print(n)
#for n,p in model.model_weights():
#    print(n)