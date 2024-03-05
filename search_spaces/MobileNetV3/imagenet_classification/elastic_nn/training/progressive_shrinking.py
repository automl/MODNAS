# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn as nn
import time
import torch
from tqdm import tqdm
import numpy as np
from timm.utils import accuracy
from search_spaces.MobileNetV3.lib.utils import MetricLogger
from search_spaces.MobileNetV3.utils import AverageMeter
from optimizers.mgd import MinNormSolver as MGD
from search_spaces.MobileNetV3.utils import (
    DistributedMetric,
    list_mean,
    val2list,
    MyRandomResizedCrop,
)
from hypernetworks.models.hpn_ofa import convert_to_dict
from search_spaces.MobileNetV3.imagenet_classification.manager import (
    DistributedRunManager
)
import torch.distributed as dist

import wandb

__all__ = [
    "validate",
    "train_one_epoch",
    "train",
    "load_models"
]


def validate(
    run_manager,
    epoch=0,
    is_test=False,
    image_size=None,
    ks=None,
    expand_ratio=None,
    depth=None,
):
    dynamic_net = run_manager.net
    dynamic_hpn = run_manager.hpn_net
    ddp_dynamic_net = run_manager.ddp_network
    ddp_hypernetwork = run_manager.ddp_hypernetwork

    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size is None:
        image_size = max(
            val2list(run_manager.run_config.data_provider.image_size, 1)
        )
    if ks is None:
        ks = max(dynamic_net.ks_list)
    if expand_ratio is None:
        expand_ratio = max(dynamic_net.expand_ratio_list)
    if depth is None:
        depth = max(dynamic_net.depth_list)

    snet_name = f"R{image_size}-D{depth}-E{expand_ratio}-K{ks}"

    supernet_settings = {
        "image_size": image_size,
        "d": depth,
        "e": expand_ratio,
        "ks": ks,
    }

    valid_log = ""

    run_manager.write_log(
        "-" * 30 + " Validate %s " % snet_name + "-" * 30, "train", should_print=False
    )
    run_manager.run_config.data_provider.assign_active_img_size(
        supernet_settings.pop("image_size")
    )
    #dynamic_net.set_active_subnet(**supernet_settings)
    run_manager.write_log(dynamic_net.module_str, "train", should_print=False)

    run_manager.reset_running_statistics(dynamic_net, dynamic_hpn)
    loss, (top1, top5) = run_manager.distributed_validate(
        epoch=epoch, is_test=is_test, run_str=snet_name, net=ddp_dynamic_net, hpn = ddp_hypernetwork,
    )

    valid_log += "%s (%.3f), " % (snet_name, top1)

    return (
        loss,
        top1,
        top5,
        valid_log,
    )



def optimize_model_weights(args, images, labels, scalarization, hw_embed, run_manager, arch_params, accum_iter, device, list_of_gradients):

    cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    # forward and backward pass
    scalarization = torch.squeeze(scalarization)
    #print(device)
    output, ce_loss, latency_loss = run_manager.ddp_network(images, labels, hw_embed, arch_params, device=device)
    #print(predictor_inp)
    loss_vector = torch.tensor([ce_loss, latency_loss]).cuda()
    cosine_similarities = cosine_sim(
                loss_vector.unsqueeze(0), scalarization.unsqueeze(0))
    #print(ce_loss)
    #print(latency_loss)
    loss = scalarization[0]*ce_loss + latency_loss*((scalarization[1]))- 0.01*cosine_similarities
    #print("Loss now", loss)
    if args.grad_scheme == "mgd":
        loss = loss/accum_iter
        loss.backward()
        grad = [param.grad.detach().clone()*accum_iter for param in
                            run_manager.ddp_hypernetwork.parameters() if param.grad is not None]
        gn = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grad]))
        # check if gn is zero
        #if gn>1e-3:
        for gr_i in range(len(grad)):
            grad[gr_i] /= gn 
        list_of_gradients.append(grad)  
        

    else:
        loss = loss/accum_iter
        loss.backward()

    run_manager.optimizer_arch.zero_grad(set_to_none=True)

    return output, loss, list_of_gradients

def optimize_simultaneous(args, arch_params, images, labels, scalarization, hw_embed, run_manager, accum_iter, current_iter, device, list_of_gradients):
    output, loss, list_of_gradients = optimize_model_weights(args, images, labels, scalarization, hw_embed, run_manager, arch_params, accum_iter, device, list_of_gradients)

    if current_iter+1 == accum_iter:
        if args.grad_scheme == "mgd":
            #print(list_of_gradients)
            gammas, _ = MGD.find_min_norm_element_FW(list_of_gradients)
            mgrad = [torch.zeros_like(x) for x in list_of_gradients[0]]
            for gamma, gr_i in zip(gammas, list_of_gradients):
                gr_i = [gamma * gr for gr in gr_i]
                mgrad = list(map(lambda x, y: x + y, gr_i, mgrad))
            for param, mg in zip(run_manager.ddp_hypernetwork.parameters(), mgrad):
                param.grad = torch.zeros_like(param)
                param.grad.data.copy_(mg)
            for param in run_manager.ddp_hypernetwork.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
            run_manager.optimizer_arch.step()
            run_manager.optimizer_arch.zero_grad(set_to_none=True)
            #run_manager.optimizer.step()
            run_manager.optimizer.zero_grad(set_to_none=True)
        else:        
            run_manager.optimizer.step()
            run_manager.optimizer_arch.step()
            run_manager.optimizer.zero_grad(set_to_none=True)
            run_manager.optimizer_arch.zero_grad(set_to_none=True)

    return output, loss, list_of_gradients


def get_current_best_resolution(arch_param_resolution, resolution_choices):
    # get the current best resolution in torch
    best_resolution = resolution_choices[torch.argmax(arch_param_resolution)]

    return best_resolution

def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0, simultaneous_opt=False):

    dynamic_net = run_manager.ddp_network
    
    distributed = isinstance(run_manager, DistributedRunManager)
    test_device = ["titan_rtx_64"]
    train_devices = ["2080ti_1","2080ti_32","2080ti_64","titan_xp_32","titan_rtx_32","titan_xp_1","titan_rtx_1","titan_xp_64","v100_1","v100_32", "v100_64"]
    p = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*2))
    # switch to train mode
    dynamic_net.train()

    if distributed:
       run_manager.run_config.train_loader.sampler.set_epoch(epoch)

    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)*len(train_devices)

    data_time = AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    metric_logger = MetricLogger(delimiter="  ")
    accum_iter = len(train_devices)#//args.world_size


    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, batch in enumerate(run_manager.run_config.train_loader):
            #hw_embed, _, _, _, _, _ = run_manager.help_loader.get_task(device)
            # shuffle the train devices
            #train_devices = list(np.random.permutation(train_devices))
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)
            #scalarization = p.sample().cuda()

            #scalarization = torch.squeeze(scalarization).unsqueeze(0)

            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )


            run_manager.optimizer.zero_grad()
            run_manager.optimizer_arch.zero_grad()
            #if simultaneous_opt is True:
            images , labels = batch
            list_of_gradients = list()
            for j in range(accum_iter):
                if args.grad_scheme == "mean":
                   run_manager.ddp_network.require_backward_grad_sync = (j == accum_iter - 1)
                   run_manager.ddp_hypernetwork.require_backward_grad_sync = (j == accum_iter - 1)
                else:
                   run_manager.ddp_network.require_backward_grad_sync = (j == accum_iter - 1)
                   run_manager.ddp_hypernetwork.require_backward_grad_sync = False

                if args.grad_scheme == "mgd":
                    run_manager.optimizer_arch.zero_grad()
                scalarization = p.sample().cuda()
                scalarization = torch.squeeze(scalarization).unsqueeze(0)
                #dist.broadcast(scalarization,0)

                hw_embed, _, _, _, _, _ = run_manager.help_loader.get_task(train_devices[j])
                hw_embed = hw_embed.cuda()
                hw_embed = torch.squeeze(hw_embed).unsqueeze(0)
                arch_params = run_manager.ddp_hypernetwork(scalarization, hw_embed)
                arch_params = convert_to_dict(arch_params)
                # interpolate the images to the best resolution
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                output, loss, list_of_gradients = optimize_simultaneous(args, arch_params, images, labels, scalarization, hw_embed, run_manager, accum_iter, j, train_devices[j], list_of_gradients)
                #print(list_of_gradients)
                # delete the images and labels
                #del images_curr
                #break
                #else:
                #    output, loss = optimizer_alternating(images, labels, images_search, labels_search, run_manager, args, epoch)

                run_manager.update_metric(metric_dict, output, labels)
                metric_logger.update(loss=loss.item())

                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                batch_size = images.shape[0]

                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                t.set_postfix(
                {
                    "loss": metric_logger.loss.avg,
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": "ce",
                    "data_time": data_time.avg,
                })
                t.update(1)
            end = time.time()
            if i == 500:
                break
    if not distributed or run_manager.is_root:
        state = {
                    "train_acc1": metric_logger.acc1.avg,
                    "train_acc5": metric_logger.acc5.avg,
                    "train_loss": metric_logger.loss.avg,
                    "epoch": epoch+1
                }

        wandb.log(state)

    run_manager.ddp_network.module.reset_ce_stats()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return (
        metric_logger.loss.avg,
        [metric_logger.acc1.avg, metric_logger.acc5.avg]
    )

def save_model(run_manager, epoch, is_best=False, curr_acc=0, best_acc=0):
    run_manager.save_model(
        {
            "epoch": epoch,
            "curr_acc": curr_acc,
            "best_acc": best_acc,
            "optimizer": run_manager.optimizer.state_dict(),
            "hpn_optimizer": run_manager.optimizer_arch.state_dict(),
            "state_dict": run_manager.network.state_dict(),
            "state_dict_hpn": run_manager.hpn_network.state_dict(),
        },
        is_best=is_best,
        model_name=run_manager.network.model_name,
        epoch=epoch
    )

def train(run_manager, args, validate_func=None):
    distributed = isinstance(run_manager, DistributedRunManager)
    # load ofa supernet
    run_manager.ddp_network.module.load_state_dict(torch.load("/path/to/ofa_mbv3_d234_e346_k357_w1.2")["state_dict"])
    if validate_func is None:
        validate_func = validate

    if args.one_shot_opt == "gdas" or args.one_shot_opt == "reinmax" or args.one_shot_opt == "reinmax2":
        run_manager.network.sampler.set_taus(0.1,10)
        run_manager.network.sampler.set_total_epochs(run_manager.run_config.n_epochs + args.warmup_epochs)

    save_model(run_manager, -1, is_best=False, curr_acc=0, best_acc=0)
    #print(run_manager.train_criterion)

    for epoch in range(
        run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        if args.one_shot_opt == "gdas" or args.one_shot_opt == "reinmax" or args.one_shot_opt == "reinmax2":
            run_manager.network.sampler.before_epoch()

        train_loss, (train_top1, train_top5) = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr, args.opt_strategy == "simultaneous"
        )

        if (epoch + 1) % 10000000 == 0:
            val_loss, val_acc1, val_acc5, _val_log = validate_func(
                run_manager, epoch=epoch, is_test=False
            )

            run_manager.best_acc = max(run_manager.best_acc, val_acc1)

            if not distributed or run_manager.is_root:
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - args.warmup_epochs,
                        run_manager.run_config.n_epochs,
                        val_loss,
                        val_acc1,
                        run_manager.best_acc,
                    )
                )
                val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(
                    top1=train_top1, loss=train_loss
                )
                val_log += _val_log
                run_manager.write_log(val_log, "valid", should_print=False)
            run_manager.best_acc = max(run_manager.best_acc, train_top1)
            if not distributed or run_manager.is_root:
                state = {
                    "train_acc1": train_top1,
                    "train_acc5": train_top5,
                    "train_loss": train_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "val_loss": val_loss,
                    "epoch": epoch+1
                }

                wandb.log(state)

        save_model(
            run_manager,
            epoch,
            is_best=train_top1==run_manager.best_acc,
            curr_acc=train_top1,
            best_acc=run_manager.best_acc
        )


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    run_manager.write_log("Loaded init from %s" % model_path, "valid")