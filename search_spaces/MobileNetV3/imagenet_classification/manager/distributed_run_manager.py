# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from predictors.help.loader import Data
from tqdm import tqdm
from utils import circle_points
from search_spaces.MobileNetV3.utils import (
    cross_entropy_with_label_smoothing,
    cross_entropy_loss_with_soft_target,
    write_log,
    init_models,
)
from torch.distributed.optim import DistributedOptimizer
from search_spaces.MobileNetV3.utils import (
    DistributedMetric,
    list_mean,
    get_net_info,
    accuracy,
    AverageMeter,
    mix_labels,
    mix_images,
)
from hypernetworks.models.hpn_ofa import convert_to_dict
from search_spaces.MobileNetV3.utils import MyRandomResizedCrop
from search_spaces.MobileNetV3.lib.utils import MetricLogger
import matplotlib.pyplot as plt

__all__ = ["DistributedRunManager"]
def plot_pareto_frontier(Xs, Ys, save_path, maxX=False, maxY=False, color="blue", xlabel="", ylabel="", legend=""):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(
        len(Xs))], reverse=maxY, key=lambda element: (element[0], element[1]))
    pareto_front = [[sorted_list[0][0], sorted_list[0][1]]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)

        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''

    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    plt.scatter(pf_X, pf_Y, color=color)
    plt.plot(pf_X, pf_Y, color=color, label=legend)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.legend()
    plt.savefig(save_path)
    plt.clf()

def preprocess(key):
    if "conv.point_linear.bn" in key:
        k_split = key.split('.')
        new_key = ''
        k_prev = ''
        i = 0
        if "bn.bn" not in key:
         for k_sub in k_split:
            if k_prev=="bn":
                if i>0:
                   new_key = new_key+".bn."+k_sub
                else:
                   new_key = k_sub
            else:
                if i>0:
                    new_key = new_key+"."+k_sub
                else:
                    new_key = k_sub
            k_prev = k_sub
            i = i+1
        else:
            print(k_split)
            del k_split[-2]
            i = 0
            for k_sub in k_split:
                if i == 0:
                    new_key = k_sub
                else:
                    new_key = new_key+'.'+k_sub
                i = i+1
        #print(new_key)
        return new_key
    return k
class DistributedRunManager:
    def __init__(
        self,
        path,
        net,
        hypernetwork,
        run_config,
        is_root=False,
        init=True,
        args=None,
    ):

        self.path = path
        self.ddp_net = net
        self.ddp_hypernetwork = hypernetwork
        self.net = net.module
        self.hpn_net = hypernetwork.module
        self.run_config = run_config
        self.is_root = is_root

        self.best_acc = 0.0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        self.net.cuda()
        cudnn.benchmark = True
        if init and self.is_root:
            init_models(self.net, self.run_config.model_init)
        if self.is_root:
            # print net info
            if args.one_shot_opt == "gdas" or args.one_shot_opt == "reinmax" or args.one_shot_opt == "reinmax2":
                self.net.sampler.set_taus(0.1,10)
                self.net.sampler.set_total_epochs(args.n_epochs + args.warmup_epochs)
                self.net.sampler.before_epoch()
            net_info = get_net_info(self.net, self.run_config.data_provider.data_shape)
            with open("%s/net_info.txt" % self.path, "w") as fout:
                fout.write(json.dumps(net_info, indent=4) + "\n")
                try:
                    fout.write(self.net.module_str + "\n")
                except Exception:
                    fout.write("%s do not support `module_str`" % type(self.net))
                #fout.write(
                #    "%s\n" % self.run_config.data_provider.train.dataset.transform
                #)
                fout.write(
                    "%s\n" % self.run_config.data_provider.test.dataset.transform
                )
                fout.write("%s\n" % self.net)

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = (
                lambda pred, target: cross_entropy_with_label_smoothing(
                    pred, target, self.run_config.label_smoothing
                )
            )
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            net_params = [
                self.net.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.net.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)
        self.optimizer_arch = torch.optim.Adam(self.ddp_hypernetwork.parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.help_loader =   Data("meta-train","datasets/help/ofa/","ofa", [], [], ["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def ddp_network(self):
        return self.ddp_net

    @ddp_network.setter
    def ddp_network(self, new_val):
        self.ddp_net = new_val

    @property
    def network(self):
        return self.net
    
    @property
    def hpn_network(self):
        return self.hpn_net

    @network.setter
    def network(self, new_val):
        self.net = new_val

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        if self.is_root:
            write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save & load model & save_config & broadcast """

    def save_config(self, extra_run_config=None, extra_net_config=None):
        if self.is_root:
            run_save_path = os.path.join(self.path, "run.config")
            if not os.path.isfile(run_save_path):
                run_config = self.run_config.config
                if extra_run_config is not None:
                    run_config.update(extra_run_config)
                json.dump(run_config, open(run_save_path, "w"), indent=4)
                print("Run configs dump to %s" % run_save_path)

            try:
                net_save_path = os.path.join(self.path, "net.config")
                net_config = self.net.config
                if extra_net_config is not None:
                    net_config.update(extra_net_config)
                json.dump(net_config, open(net_save_path, "w"), indent=4)
                print("Network configs dump to %s" % net_save_path)
            except Exception:
                print("%s do not support net config" % type(self.net))

    def save_model(self, checkpoint=None, is_best=False, model_name=None, epoch=0):
        if self.is_root:
            if checkpoint is None:
                checkpoint = {"state_dict": self.net.state_dict()}

            if model_name is None:
                model_name = f"checkpoint_e{epoch}.pth.tar"
            else:
                model_name += f"_e{epoch}.pth.tar"

            latest_fname = os.path.join(self.save_path, "latest.txt")
            model_path = os.path.join(self.save_path, model_name)
            with open(latest_fname, "w") as _fout:
                _fout.write(model_path + "\n")
            torch.save(checkpoint, model_path)

            if is_best:
                best_path = os.path.join(self.save_path, f"best_{model_name}")
                torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

    def load_model(self, model_fname=None):
            #if self.is_root:
            #latest_fname = os.path.join(self.save_path, "latest.txt")
            #if model_fname is None and os.path.exists(latest_fname):
            #    with open(latest_fname, "r") as fin:
            #        model_fname = fin.readline()
            #        if model_fname[-1] == "\n":
            #            model_fname = model_fname[:-1]
            # noinspection PyBroadException
            #try:
            #    if model_fname is None or not os.path.exists(model_fname):
            #        model_fname = "%s/checkpoint.pth.tar" % self.save_path
            #        with open(latest_fname, "w") as fout:
            #            fout.write(model_fname + "\n")
            #    print("=> loading checkpoint '{}'".format(model_fname))
            #    checkpoint = torch.load(model_fname, map_location="cpu")
            #except Exception:
            #    self.write_log(
            #        "fail to load checkpoint from %s" % self.save_path, "valid"
            #    )
            #    return
            state_dict = torch.load("path/to/model")["state_dict"]
            state_dict_new = {}
            for k in state_dict.keys(): 
                if "arch" not in k:
                    if k not in self.net.state_dict():
                        #print("Key before", k)
                        k_new = preprocess(k)
                        #print("Key after", k)
                        state_dict_new[k_new] = state_dict[k]
                    else:
                        state_dict_new[k] = state_dict[k]
                else:
                    print(k)
                    print(state_dict[k])
            list_ours = list(state_dict_new.keys())
            list_ofa = list(self.net.state_dict().keys())
            print(sorted(list(set(list_ofa) - set(list_ours))))
            print(sorted(list(set(list_ours) - set(list_ofa))))
            self.net.load_state_dict(state_dict_new)
            print("Loaded state dict")
            self.net.set_best_net()
            #subnet = self.net.get_active_subnet(preserve_weight=True)
            #self.net = subnet
            #self.net.load_state_dict(checkpoint["state_dict"])
            #if "epoch" in checkpoint:
            #    self.start_epoch = checkpoint["epoch"] + 1
            #if "best_acc" in checkpoint:
            #    self.best_acc = checkpoint["best_acc"]
            #if "optimizer" in checkpoint:
            #    self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.write_log("=> loaded checkpoint '{}'".format(model_fname), "valid")

    # noinspection PyArgumentList
    def broadcast(self):

        self.start_epoch = torch.LongTensor(1).fill_(self.start_epoch)[0]
        self.best_acc = torch.Tensor(1).fill_(self.best_acc)[0]

    """ metric related """

    def get_metric_dict(self):
        return {
            "top1": DistributedMetric("top1"),
            "top5": DistributedMetric("top5"),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0], output.size(0))
        metric_dict["top5"].update(acc5[0], output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg.item() for key in metric_dict}
        else:
            return [metric_dict[key].avg.item() for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"

    """ train & validate """

    def validate(
        self,
        epoch=0,
        is_test=False,
        run_str="",
        net=None,
        hpn = None,
        data_loader=None,
        no_logs=False,
    ):
        if net is None:
            net = self.ddp_net
        if data_loader is None:
            #if is_test:
            data_loader = self.run_config.test_loader
            #else:
            #data_loader = self.run_config.valid_loader

        net.eval()
        test_rays = circle_points(1)
        losses = DistributedMetric("val_loss")
        cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        pareto_accs  = []
        pareto_losses = []
        pareto_latencies = []
        devices_all = ["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_xp_1","titan_xp_32","titan_xp_64","v100_1","v100_32", "v100_64", "titan_rtx_64" ]
        resolution_list =  [128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224]
        for device in devices_all:
         hw_embed, _, _, _, _, _ = self.help_loader.get_task(device)
         hw_embed = torch.squeeze(hw_embed).unsqueeze(0).cuda()
         for test_ray in test_rays:
          metric_dict = self.get_metric_dict()
          losses = DistributedMetric("val_loss")
          scalarization = torch.tensor(test_ray).cuda()
          scalarization = scalarization.unsqueeze(0)
          arch_params = hpn(scalarization, hw_embed)
          best_resolution = get_current_best_resolution(arch_params[-1],resolution_list)
          with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                disable=no_logs or not self.is_root,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images = nn.functional.interpolate(images, size=(best_resolution, best_resolution), mode='bilinear', align_corners=False)
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output, ce_loss, latency_loss, _ = net(images, labels, hw_embed, arch_params)
                    #output = net(images)
                    cosine_similarities = cosine_sim(loss_vector.unsqueeze(0), scalarization)
                    loss = ce_loss*scalarization[0][0] + latency_loss*scalarization[0][1] - 0.001*cosine_similarities
                    loss_vector = torch.tensor([ce_loss, latency_loss]).cuda()
                    # measure accuracy and record loss
                    losses.update(loss, images.size(0))
                    self.update_metric(metric_dict, output, labels)
                    t.set_postfix(
                        {
                            "loss": losses.avg.item(),
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)
                    #break
            # gather the stats from all processes
            metric_dict["top1"].synchronize_between_processes()
            metric_dict["top5"].synchronize_between_processes()
            print(
                '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_dict["top1"],
                        top5=metric_dict["top5"],
                        losses=losses)
            )
            pareto_accs.append(metric_dict["top1"].avg.item())
            pareto_losses.append(losses.avg.item())
            pareto_latencies.append(latency_loss.item())
        print(pareto_accs)
        print(pareto_losses)
        print(pareto_latencies)
        return losses.avg.item(), self.get_metric_vals(metric_dict)


    @torch.no_grad()
    def distributed_validate(self,
        epoch=0,
        is_test=False,
        run_str="",
        no_logs=False,
        net=None,
        hpn=None,
        data_loader=None,
    ):

        model = self.ddp_net if net is None else net
        hpn = self.ddp_hypernetwork if hpn is None else hpn
        criterion = torch.nn.CrossEntropyLoss()
        test_rays = circle_points(10)
        losses = DistributedMetric("val_loss")
        cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        devices_all = self.help_loader.meta_train_devices+self.help_loader.meta_valid_devices+self.help_loader.meta_test_devices
        if data_loader is None:
            #if is_test:
            data_loader = self.run_config.test_loader
            #else:
            #    data_loader = self.run_config.valid_loader

        
        header = 'Test:' if is_test else 'Validation:'

        model.eval()
        hpn.eval()
        for device in devices_all:
         hw_embed, _, _, _, _, _ = self.help_loader.get_task(device)
         hw_embed = torch.squeeze(hw_embed).unsqueeze(0).cuda()
         pareto_accs = []
         pareto_latencies = []
         for test_ray in test_rays:
          metric_logger = MetricLogger(delimiter="  ")
          #metric_logger.add_meter("acc1")
          metric_logger.add_meter("latency")
          scalarization = torch.tensor(test_ray).cuda()
          scalarization = scalarization.unsqueeze(0)
          arch_params = hpn(scalarization, hw_embed)
          arch_params = convert_to_dict(arch_params)
          #representation = model.module.get_arch_represenation(arch_params)
          #archs_pareto.append(representation)
          with tqdm(
            total=len(data_loader),
            desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
            disable=no_logs or not self.is_root) as t:
            for i, (images, target) in enumerate(data_loader):
                #image = nn.functional.interpolate(images, size=(best_resolution, best_resolution), mode='bilinear', align_corners=False)
                image, labels = images.cuda(), target.cuda()
                output, ce_loss, latency_loss = model(image, labels, hw_embed, arch_params, device = device)
                
                #loss_vector = torch.tensor([ce_loss, latency_loss]).cuda()
                #cosine_similarities = cosine_sim(loss_vector.unsqueeze(0), scalarization)
                #loss = ce_loss*scalarization[0][0] + latency_loss*scalarization[0][1] - 0.001*cosine_similarities


                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                t.update(1)
                batch_size = image.shape[0]
                #metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                metric_logger.meters['latency'].update(latency_loss.item(), n=1)
                #break

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            pareto_accs.append(100-metric_logger.acc1.avg)
            pareto_latencies.append(metric_logger.latency.avg)
            ##pareto_accs.append(metric_logger.acc1.avg)
            #pareto_losses.append(metric_logger.loss.avg)
            #pareto_latencies.append(latency_loss.item())
            #print(pareto_accs)
            #print(pareto_losses)
            #print(pareto_latencies)
         save_path = "ofa_paretos/"+ device +"_"+str(epoch)+".png"
         print(pareto_accs)
         print(pareto_latencies)
         plot_pareto_frontier(pareto_accs, pareto_latencies, save_path, maxX=False, maxY=False, color="blue", xlabel="Error", ylabel="Latency", legend=device)
         
        #for p in model.module.arch_parameters():
        #    print(torch.nn.functional.softmax(p,dim=-1))
        #print(archs_pareto)
        return (
            0,
            [metric_logger.acc1.avg, 0]
        )

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.net
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, (top1, top5) = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            loss, (top1, top5) = self.validate(epoch, is_test, net=net)
            return (
                [self.run_config.data_provider.active_img_size],
                [loss],
                [top1],
                [top5],
            )

    def train_one_epoch(self, args, epoch, warmup_epochs=5, warmup_lr=0):
        self.net.train()
        #self.run_config.train_loader.sampler.set_epoch(
        #    epoch
        #)  # required by distributed sampler
        MyRandomResizedCrop.EPOCH = epoch  # required by elastic resolution

        nBatch = len(self.run_config.train_loader)

        losses = DistributedMetric("train_loss")
        metric_dict = self.get_metric_dict()
        data_time = AverageMeter()
        print(len(self.run_config.train_loader))
        with tqdm(
            total=nBatch,
            desc="Train Epoch #{}".format(epoch + 1),
            disable=not self.is_root,
        ) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                MyRandomResizedCrop.BATCH = i
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer,
                        warmup_epochs * nBatch,
                        nBatch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(
                        self.optimizer, epoch - warmup_epochs, i, nBatch
                    )

                images, labels = images.cuda(), labels.cuda()
                target = labels
                if isinstance(self.run_config.mixup_alpha, float):
                    # transform data
                    random.seed(int("%d%.3d" % (i, epoch)))
                    lam = random.betavariate(
                        self.run_config.mixup_alpha, self.run_config.mixup_alpha
                    )
                    images = mix_images(images, lam)
                    labels = mix_labels(
                        labels,
                        lam,
                        self.run_config.data_provider.n_classes,
                        self.run_config.label_smoothing,
                    )

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                output = self.ddp_net(images)

                if args.teacher_model is None:
                    loss = self.train_criterion(output, labels)
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(
                            output, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + self.train_criterion(
                        output, labels
                    )
                    loss_type = "%.1fkd+ce" % args.kd_ratio

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(loss, images.size(0))
                self.update_metric(metric_dict, output, target)

                t.set_postfix(
                    {
                        "loss": losses.avg.item(),
                        **self.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                        "lr": new_lr,
                        "loss_type": loss_type,
                        "data_time": data_time.avg,
                    }
                )
                t.update(1)
                end = time.time()
                print(losses.avg.item())
                #break
        return losses.avg.item(), self.get_metric_vals(metric_dict)

    def train(self, args, warmup_epochs=5, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epochs):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(
                args, epoch, warmup_epochs, warmup_lr
            )
            img_size, val_loss, val_top1, val_top5 = self.validate_all_resolution(
                epoch, is_test=False
            )

            is_best = list_mean(val_top1) > self.best_acc
            self.best_acc = max(self.best_acc, list_mean(val_top1))
            if self.is_root:
                val_log = (
                    "[{0}/{1}]\tloss {2:.3f}\t{6} acc {3:.3f} ({4:.3f})\t{7} acc {5:.3f}\t"
                    "Train {6} {top1:.3f}\tloss {train_loss:.3f}\t".format(
                        epoch + 1 - warmup_epochs,
                        self.run_config.n_epochs,
                        list_mean(val_loss),
                        list_mean(val_top1),
                        self.best_acc,
                        list_mean(val_top5),
                        *self.get_metric_names(),
                        top1=train_top1,
                        train_loss=train_loss
                    )
                )
                for i_s, v_a in zip(img_size, val_top1):
                    val_log += "(%d, %.3f), " % (i_s, v_a)
                self.write_log(val_log, prefix="valid", should_print=False)

                self.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": self.best_acc,
                        "optimizer": self.optimizer.state_dict(),
                        "state_dict": self.net.state_dict(),
                        "state_dict_archs": self.net.state_dict_archs(),
                    },
                    is_best=is_best,
                )

    def reset_running_statistics(
        self, net=None, hpn=None, subset_size=2000, subset_batch_size=200, data_loader=None
    ):
        from search_spaces.MobileNetV3.imagenet_classification.elastic_nn.utils import set_running_statistics

        if net is None:
            net = self.net
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(
                subset_size, subset_batch_size
            )
        set_running_statistics(net, hpn,  data_loader)
        del data_loader

def get_current_best_resolution(arch_param_resolution, resolution_choices):
    # get the current best resolution in torch
    best_resolution = resolution_choices[torch.argmax(arch_param_resolution)]

    return best_resolution