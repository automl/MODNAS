# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
from predictors.help.loader import Data
from search_spaces.MobileNetV3.utils import AverageMeter, get_net_device, DistributedTensor
from search_spaces.MobileNetV3.modules.dynamic_op import DynamicBatchNorm2dMixture
from hypernetworks.hpns_ofa import convert_to_dict
__all__ = ["set_running_statistics"]


def set_running_statistics(model, hpn, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    forward_hpn = copy.deepcopy(hpn)
    set_running_statistics_flag = True
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if distributed:
                bn_mean[name] = DistributedTensor(name + "#mean")
                bn_var[name] = DistributedTensor(name + "#var")
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = (
                        x.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    with torch.no_grad():
        DynamicBatchNorm2dMixture.SET_RUNNING_STATISTICS = True
        p = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*2))
        devices_all = ["titan_rtx_64" ]
        help_loader =  Data("meta-train","datasets/help/ofa/","ofa", meta_test_devices=[], meta_valid_devices=[], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )
        for images, labels in data_loader:
            images = images.to(get_net_device(forward_model))
            labels = labels.to(get_net_device(forward_model))
            scalarization = p.sample()
            scalarization = scalarization.to(get_net_device(forward_model))
            scalarization = scalarization.unsqueeze(0)
            for device in devices_all:
                hw_embed, _, _, _, _, _ = help_loader.get_task(device=device)
                hw_embed = hw_embed.to(get_net_device(forward_model)).unsqueeze(0)
                #arch_params = 
                forward_model(images,labels,hw_embed, convert_to_dict(hpn(scalarization, hw_embed)), device=device, set_running_statistics=set_running_statistics_flag)
        DynamicBatchNorm2dMixture.SET_RUNNING_STATISTICS = False
        set_running_statistics_flag = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
