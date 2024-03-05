from hypernetworks.models.hpn_ofa import MetaHyperNetwork, HyperNetwork
from predictors.help.loader import Data
from search_spaces.MobileNetV3.model_search_ours import OFAMobileNetV3Mixture
import argparse
import torch
import os
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="depth",
    choices=[
        "kernel",
        "depth",
        "expand",
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")
parser.add_argument("--one_shot_opt", type=str, default="reinmax")
parser.add_argument("--opt_strategy", type=str, default="simultaneous")
parser.add_argument("--valid_size", type=float, default=0.8)
parser.add_argument("--hpn_type", type=str, default="meta-2")

args = parser.parse_args()
if args.task == "kernel":
    args.path = "exp/normal2kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "depth":
    args.path = "exp/kernel2kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "2,3,4"
args.n_epochs = 410
args.base_lr = 7.5e-3
args.warmup_epochs = 5
args.warmup_lr = -1
args.ks_list = "3,5,7"
args.expand_list = "3,4,6"
args.depth_list = "2,3,4"
args.manual_seed = 0
args.path = "experiments/mobilenet_"+args.one_shot_opt+"_"+args.opt_strategy
args.seed = 9001
args.lr_schedule_type = "cosine"
args.world_size = 1
args.base_batch_size = 256

if args.opt_strategy == "simultaneous":
    args.valid_size = None

args.arch_learning_rate = 3e-4
args.arch_weight_decay = 1e-3 
args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10
args.dist_url = "env://"
args.n_worker = 0
args.resize_scale = 0.08
args.distort_color = "tf"
args.image_size = "128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224"
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = "proxyless"

args.width_mult_list = "1.2"
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = "ce"
args.device = "cuda"

args.save_path = "ofa_hpn/"
choices  = {}
choices["kernel_size"] = [3,5,7]
choices["expand_ratio"] = [3,4,6]
choices["depth"] = [2,3,4]
args.width_mult_list = [
        float(width_mult) for width_mult in args.width_mult_list.split(",")
    ]
args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
args.expand_list = [int(e) for e in args.expand_list.split(",")]
args.depth_list = [int(d) for d in args.depth_list.split(",")]

args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )
help_loader =  Data(mode="meta-train",data_path="datasets/help/ofa",search_space="ofa", meta_valid_devices=[], meta_test_devices= [], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='')
hw_embeddings = []
for device in ["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"]:
    hw_embeddings.append(help_loader.get_task(device)[0].cuda())
net = OFAMobileNetV3Mixture(
        n_classes=1000,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        base_stage_width=args.base_stage_width,
        width_mult=args.width_mult_list,
        ks_list=args.ks_list,
        expand_ratio_list=args.expand_list,
        depth_list=args.depth_list,
        optimizer = args.one_shot_opt,
    )
hypernetwork1 = MetaHyperNetwork(len(net.block_group_info),max_depth=max(net.depth_list), kernel_list_len=len(net.ks_list), expand_ratio_list_len=len(net.expand_ratio_list), depth_list_len=len(net.depth_list), resolution_len=len(net.resolution_list),hpn_type=HyperNetwork, use_zero_init=False,use_softmax=False)
hypernetwork2 = MetaHyperNetwork(len(net.block_group_info),max_depth=max(net.depth_list), kernel_list_len=len(net.ks_list), expand_ratio_list_len=len(net.expand_ratio_list), depth_list_len=len(net.depth_list), resolution_len=len(net.resolution_list),hpn_type=HyperNetwork, use_zero_init=False,use_softmax=True)
hpn_list = [hypernetwork1]
hpn_names = ["ofa_metahpn"]
for hpn, name in zip(hpn_list, hpn_names):
    hpn.cuda()
    p = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*2))
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(hpn.parameters(), lr=1e-3)
    os.makedirs("pretrained_hpns", exist_ok=True)
    for i in range(10):
      for j in range(1000):
        scalarization = p.sample().cuda().unsqueeze(0)
        hw_emb = help_loader.sample_device_embedding(mode="meta_train")[0].unsqueeze(0).cuda()
        out_kernel = 1e-3*torch.randn([len(net.block_group_info),max(net.depth_list),len(net.ks_list)]).cuda()
        out_expand = 1e-3*torch.randn([len(net.block_group_info),max(net.depth_list),len(net.expand_ratio_list)]).cuda()
        out_depth = 1e-3*torch.randn([len(net.block_group_info),len(net.depth_list)]).cuda()
        out_resolution = 1e-3*torch.randn([len(net.resolution_list)]).cuda()
        pred_kernel, pred_expand, pred_depth, pred_resolution = hpn(scalarization, hw_emb)
        loss = mse_loss(pred_kernel,out_kernel) + mse_loss(pred_expand, out_expand) + mse_loss(pred_depth, out_depth) + mse_loss(out_resolution, pred_resolution)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if j%100==0:
            print("Loss:", loss.item())
            print(pred_kernel.softmax(dim=-1))
            print(pred_expand.softmax(dim=-1))
            print(pred_depth.softmax(dim=-1))
            print(pred_resolution.softmax(dim=-1))

            torch.save(hpn.state_dict(), "pretrained_hpns/"+name+".pth")
