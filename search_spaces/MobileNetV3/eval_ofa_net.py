# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from search_spaces.MobileNetV3.ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from search_spaces.MobileNetV3.ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from search_spaces.MobileNetV3.ofa.model_zoo import ofa_net


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of imagenet", type=str, default="path/to/imagenet"
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=100,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
parser.add_argument(
    "-n",
    "--net",
    metavar="OFANET",
    default="ofa_resnet50",
    choices=[
        "ofa_mbv3_d234_e346_k357_w1.0",
        "ofa_mbv3_d234_e346_k357_w1.2",
        "ofa_proxyless_d234_e346_k357_w1.3",
        "ofa_resnet50",
    ],
    help="OFA networks",
)

args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path
run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)
import pickle
with open("path/to/modnas_archs.pkl","rb") as f:
     help_results = pickle.load(f)
gt_results = {}
for device in help_results:
    archs = help_results[device]
    gt_results[device] = {}
    for k in archs:
       arch = archs[k]
       if arch!=None:
        ofa_network = ofa_net(args.net, pretrained=True)
        """ Randomly sample a sub-network, 
        you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
        """
        ofa_network.set_active_subnet(ks=arch["ks"], e=arch["e"], d=arch["d"]) 
        #ofa_network.sample_active_subnet()
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        """ Test sampled subnet """
        run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
        # assign image size: 128, 132, ..., 224
        run_config.data_provider.assign_active_img_size(arch["r"])
        run_manager.reset_running_statistics(net=subnet)
        print("Test random subnet:")
        print(subnet.module_str)
        loss, (top1, top5) = run_manager.validate(net=subnet)
        print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))
        gt_results[device][k] = (loss, top1, top5, arch["ks"], arch["e"], arch["d"], arch["r"])
        with open("modnas_results.pkl", "wb") as f:
             pickle.dump(gt_results, f)