from utils import create_exp_dir, count_parameters_in_MB, \
_data_transforms_cifar10, _data_transforms_cifar100, _data_transforms_svhn, \
AvgrageMeter, accuracy, analysis, CustomCosineAnnealing
from hypernetworks.models.hpn_nb201 import MetaHyperNetwork
from search_spaces.nb201.model_search import NASBench201SearchSpace
from predictors.help.loader import Data
from optimizers.mgd import MinNormSolver as MGD

from utils import circle_points
from predictors.help.utils import denorm
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torch.utils
import torch.nn as nn
import argparse
import logging
import torch
import numpy as np
import random
import glob
import time
import os
import sys
import wandb

sys.path.insert(0, '../')


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='datapath',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str,
                    default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float,
                    default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='nasbench201', help='experiment name')
parser.add_argument('--wandb_name', type=str,
                    default='modnas-nb201-intervals-fixnorm-ceweight', help='optimizer type')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=0.001, help='weight decay for arch encoding')
parser.add_argument('--tau_max', type=float, default=10,
                    help='Max temperature (tau) for the gumbel softmax.')
parser.add_argument('--tau_min', type=float, default=1,
                    help='Min temperature (tau) for the gumbel softmax.')
parser.add_argument('--k', type=int, default=1,
                    help='partial channel parameter')
parser.add_argument('--num_objectives', type=int,
                    default=2, help='number of objectives')
parser.add_argument('--num_test', type=int, default=24,
                    help='number of test scalarizations to sample')
parser.add_argument('--use_we_v2', action='store_true',
                    default=False, help='use we_v2')
parser.add_argument('--use_zero_init', action='store_true',
                    default=False, help='use zero init in HPN')
parser.add_argument('--optimizer_type', type=str,
                    default='reinmax2', help='optimizer type')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='number of warmup epochs')
# regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=[
                    'l2', 'kl'], help='regularization type, kl is implemented for dirichlet only')
parser.add_argument('--reg_scale', type=float, default=1e-3,
                    help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')
parser.add_argument('--latency_norm_scheme', type=str, default='predictor_stats')
parser.add_argument('--hpn_help_features', action='store_true', default=False,
                   help='use the HELP features in the HPN or just the device id')
# hardware embedding
parser.add_argument('--hw_embed_dim', type=int, default=10, help="the dimension of hardware embedding")
parser.add_argument('--layer_size', type=int, default=100, help="the size of hidden layer of the predictor")
parser.add_argument('--num_random_devices', type=int, default=50, help="the dimension of hardware embedding")
parser.add_argument('--load_path', type=str,
                    default='predictor_data_utils/predictor_meta_learned.pth', help='model checkpoint path')
parser.add_argument('--w_grad_update_method', type=str,
                    default="mean", help='update method for oneshot weights')
parser.add_argument('--hpn_grad_update_method', type=str,
                    default="mgd", help='update method for hypernet weights')
parser.add_argument('--hpn_s1_scaling', action='store_true',
                    default=False, help='use hypernet learning rate scheduler')
parser.add_argument('--w_s1_scaling', action='store_true',
                    default=False, help='use hypernet learning rate scheduler')
parser.add_argument('--use_cosine_sim_schedule', action='store_true',
                    default=False, help='use cosine simlarity schedule')
parser.add_argument('--num_intervals', type=int, default=101, help='number of intervals in hpn')
parser.add_argument('--ce_weight', type=float, default=1., help='CE weight in the Dirichlet dist')
parser.add_argument('--fix_norm', action='store_true',
                    default=False, help='scale latency to the CE range')
parser.add_argument('--lat_const_normalized', type=float, default=0.,
                    help='normalized latency constrain')
parser.add_argument('--no_grad_norm', action='store_true',
                    default=False, help='normilize mgd gradient')
args = parser.parse_args()


args.save = \
f'experiments_mgd_final/{args.save}/search-{time.strftime("%Y%m%d-%H%M%S")}-{args.seed}-{args.optimizer_type}-{args.weight_decay}-{args.use_we_v2}-{args.epochs}-{args.train_portion}-{args.num_random_devices}-{args.tau_max}-{args.tau_min}-{args.w_grad_update_method}-{args.hpn_grad_update_method}-{args.arch_weight_decay}-{args.learning_rate}-{args.learning_rate_min}-{args.hpn_s1_scaling}-{args.w_s1_scaling}-{args.use_cosine_sim_schedule}-{args.num_intervals}-{args.ce_weight}-{args.fix_norm}-{args.lat_const_normalized}-{args.no_grad_norm}'
if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
create_exp_dir(args.save)#, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#wandb logging
project_name = args.wandb_name
name = \
f'search-{time.strftime("%Y%m%d-%H%M%S")}-{args.seed}-{args.optimizer_type}-{args.weight_decay}-{args.use_we_v2}-{args.epochs}-{args.train_portion}-{args.num_random_devices}-{args.tau_max}-{args.tau_min}-{args.w_grad_update_method}-{args.hpn_grad_update_method}-{args.arch_weight_decay}-{args.learning_rate}-{args.learning_rate_min}-{args.hpn_s1_scaling}-{args.w_s1_scaling}-{args.use_cosine_sim_schedule}-{args.num_intervals}-{args.ce_weight}-{args.fix_norm}-{args.lat_const_normalized}-{args.no_grad_norm}'
#wandb.init(project=project_name, entity="project-name", name=name)
#wandb.config.update(args.__dict__)
#wandb.config.seed = args.seed


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = NASBench201SearchSpace(16, 5, 4, 10,
                                   metric="fpga_latency", #placeholder, not used
                                   entangle_weights=True,
                                   optimizer_type=args.optimizer_type,
                                   latency_norm_scheme=args.latency_norm_scheme,
                                   hw_embed_on=True,
                                   hw_embed_dim=args.hw_embed_dim,
                                   layer_size=args.layer_size,
                                   load_path=args.load_path)

    help_loader = Data(mode='meta_test',
                       data_path='datasets/help/nasbench201',
                       search_space='nasbench201',
                       meta_train_devices=['1080ti_1',
                                           '1080ti_32',
                                           '1080ti_256',
                                           'silver_4114',
                                           'silver_4210r',
                                           'samsung_a50',
                                           'pixel3',
                                           'essential_ph_1',
                                           'samsung_s7'],
                       meta_valid_devices=['titanx_1',
                                           'titanx_32',
                                           'titanx_256',
                                           'gold_6240'],
                       meta_test_devices=['titan_rtx_256',
                                          'gold_6226',
                                          'fpga',
                                          'pixel2',
                                          'raspi4',
                                          'eyeriss'],
                       num_inner_tasks=8,
                       num_meta_train_sample=900,
                       num_sample=10,
                       num_query=1000,
                       sampled_arch_path=\
                       'datasets/help/nasbench201/arch_generated_by_metad2a.txt'
                      )
    hw_embeddings = list()
    for d in help_loader.meta_train_devices+help_loader.meta_valid_devices:
        hw_emb, _, _, _, _, _ = help_loader.get_task(d)
        hw_embeddings.append(hw_emb)



    if args.w_grad_update_method in ["mean"]:
        args.learning_rate *= len(hw_embeddings)
    if args.hpn_grad_update_method in ["mean", "mgd"]:
        args.arch_learning_rate *= len(hw_embeddings)
    hpn_name_str = "MetaHyperNetworkHyperNetworkTrue"
    use_sm = True
    hpn = MetaHyperNetwork(num_random_devices=args.num_random_devices,num_intervals=args.num_intervals)
    hpn_rand = MetaHyperNetwork(num_random_devices=args.num_random_devices)

    full_hpn_path = "hypernetwork_data_utils/nb201/"+hpn_name_str+".pth"
    hpn.load_state_dict(torch.load(full_hpn_path))
    print("Loaded pretrained hpn from ", full_hpn_path)
    hpn = hpn.cuda()
    hpn_rand = hpn_rand.cuda()
    model = model.cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_arch = torch.optim.Adam(hpn.parameters(), lr=args.arch_learning_rate, betas=(
        0.5, 0.999), weight_decay=args.arch_weight_decay)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = _data_transforms_cifar100(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = _data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train',
                               download=True, transform=train_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from nasbench201.DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(
            16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(
            args.data, 'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_arch, float(args.epochs),
        eta_min=0)

    scheduler_cosine_sim = CustomCosineAnnealing(eta_min=0.1, eta_max=0.01,
                                                 T_max=args.epochs)

    model.sampler.set_taus(args.tau_min, args.tau_max)
    model.sampler.set_total_epochs(args.epochs)
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        lr_arch = scheduler_arch.get_lr()[0]
        if args.use_cosine_sim_schedule:
            cosine_sim_coeff = scheduler_cosine_sim.get_value(epoch)
        else:
            cosine_sim_coeff = scheduler_cosine_sim.get_value(0)
        logging.info('epoch %d lr %e lr_arch %e', epoch, lr, lr_arch)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        # model.show_arch_parameters()

        # training
        model.sampler.before_epoch()
        train_acc, train_obj = train(
            args, train_queue, valid_queue, model, hpn, criterion, optimizer,
            lr, epoch, optimizer_arch, help_loader, cosine_sim_coeff)
        #wandb.log({"train_acc": train_acc, "train_loss": train_obj})
        logging.info('train_acc %f', train_acc)
        # save model
        model_path = os.path.join(args.save, 'weights.pt')
        torch.save(model.state_dict(), model_path)
        hpn_path = os.path.join(args.save, 'hpn.pt')
        torch.save(hpn.state_dict(), hpn_path)
        # validation
        test_results = infer(model, hpn, hpn_rand, args, epoch, help_loader)

        logging.info('test_results %s', test_results)
        model.reset_batch_stats()

        scheduler.step()



def train(args, train_queue, valid_queue, model, hpn, criterion, optimizer, lr,
          epoch, optimizer_arch, help_loader, cosine_sim_coeff):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    num_objs = args.num_objectives  # NOTE: make this an argument
    p = torch.distributions.dirichlet.Dirichlet(torch.tensor([args.ce_weight,
                                                              1.]))
    cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for step, (input, target) in enumerate(train_queue):
        model.train()
        optimizer_arch.zero_grad()
        optimizer.zero_grad()
        n = input.size(0)

        scalarization = p.sample().cuda()

        if epoch >= args.warmup_epochs:
            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            set_of_devices = help_loader.meta_train_devices + help_loader.meta_valid_devices
            if args.hpn_grad_update_method == 'sample':
                set_of_devices = random.sample(set_of_devices, 1)

            accum_iter = len(set_of_devices)
            list_of_gradients = list()
            logging.info('Using %s to compute hpn gradients...',
                         args.hpn_grad_update_method)
            for device_idx, device in enumerate(set_of_devices):
                if (args.hpn_grad_update_method in ['mgd', 'sample',
                                                    'sequential']):
                    optimizer.zero_grad()
                    optimizer_arch.zero_grad()
                scalarization = p.sample().cuda()
                hw_emb_search, _, _, _, _, _ = help_loader.get_task(device)
                hw_emb_search = hw_emb_search.cuda()

                # LATENCY CONSTRAINT
                device_latencies = help_loader.latency[device]
                latency_constr = denorm(args.lat_const_normalized,
                                        max(device_latencies),
                                        min(device_latencies))

                arch_params = hpn(scalarization.unsqueeze(0), hw_emb_search.unsqueeze(0))
                device_name = device +"_latency"
                logits, ce_loss, latency_loss, arch = model(
                    input_search, target_search, arch_params,
                    hw_emb_search.unsqueeze(0), hw_metric=device_name,
                    fix_norm=args.fix_norm)
                loss_vector = torch.tensor([ce_loss, latency_loss]).cuda()
                cosine_similarities = cosine_sim(
                    loss_vector.unsqueeze(0), scalarization.unsqueeze(0))
                if args.hpn_grad_update_method in ["mean", "mgd", "sequential"]:
                    if args.hpn_s1_scaling:
                        scalarization[0] /= accum_iter
                if model.latencies[-1] <= latency_constr:
                    loss_val = ce_loss
                else:
                    loss_val = ce_loss*scalarization[0] + latency_loss*scalarization[1] - cosine_sim_coeff * cosine_similarities

                if args.hpn_grad_update_method == 'mgd':
                    loss_val.backward(retain_graph=True)
                    #nn.utils.clip_grad_norm_(hpn.parameters(), args.grad_clip)
                    grad = [param.grad.detach().clone() for param in
                            hpn.parameters() if param.grad is not None]

                    if not args.no_grad_norm:
                        # normalize grads
                        gn = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grad]))
                        if not model.latencies[-1] <= latency_constr:
                          for gr_i in range(len(grad)):
                            grad[gr_i] /= gn

                    #grad = torch.nn.functional.normalize(grad, dim=0).reshape(1, -1)
                    list_of_gradients.append(grad)
                else:
                    if args.hpn_grad_update_method == 'mean':
                       loss_val /= accum_iter

                    loss_val.backward()
                    if args.hpn_grad_update_method in ['mean', 'sum']:
                       if device_idx + 1 == accum_iter:
                          #g_norm = compute_l2_norm(hpn)
                          #wandb.log({'hpn_grad_norm': g_norm})
                          optimizer_arch.step()
                          optimizer.zero_grad()
                          optimizer_arch.zero_grad()
                       else:
                          continue
                    else: # for sequential and sample
                       optimizer_arch.step()

            if args.hpn_grad_update_method == 'mgd':
                # compute the gamma coefficients that will be used to scale every
                # gradient vector corresponding to one device
                gammas, _ = MGD.find_min_norm_element_FW(list_of_gradients)

                # sum the scaled gradients
                mgrad = [torch.zeros_like(x) for x in list_of_gradients[0]]
                for gamma, gr_i in zip(gammas, list_of_gradients):
                    gr_i = [gamma * gr for gr in gr_i]
                    mgrad = list(map(lambda x, y: x + y, gr_i, mgrad))

                # update the parameters of the hypernet using MGD
                for param, mg in zip(hpn.parameters(), mgrad):
                    if param.grad is not None:
                        param.grad.data.copy_(mg)

                #g_norm = compute_l2_norm(hpn)
                #wandb.log({'hpn_grad_norm': g_norm})
                optimizer_arch.step()
                optimizer_arch.zero_grad()

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        set_of_devices = help_loader.meta_train_devices + help_loader.meta_valid_devices
        if args.w_grad_update_method == 'sample':
            set_of_devices = random.sample(set_of_devices, 1)

        accum_iter = len(set_of_devices)
        for device_idx, device in enumerate(set_of_devices):
            if (args.w_grad_update_method in ['sample', 'sequential']):
                optimizer.zero_grad()
                optimizer_arch.zero_grad()
            hw_emb, _, _, _, _, _ = help_loader.get_task(device)
            hw_emb = hw_emb.cuda()
            scalarization = p.sample().cuda()
            # obtain arch params
            device_name = device +"_latency"
            arch_params = hpn(scalarization.unsqueeze(0), hw_emb.unsqueeze(0))
            logits, ce_loss, latency_loss, arch = model(input, target,
                                                        arch_params,
                                                        hw_emb.unsqueeze(0),
                                                        hw_metric=device_name,
                                                        fix_norm=args.fix_norm)
            #loss_vector = torch.tensor([ce_loss, latency_loss]).cuda()
            #cosine_similarities = cosine_sim(
                #loss_vector.unsqueeze(0), scalarization.unsqueeze(0))
            if args.w_grad_update_method in ["mean", "sequential"]:
                if not args.w_s1_scaling:
                    scalarization[0] = 1
            loss_train = ce_loss*scalarization[0]

            if args.w_grad_update_method == 'mean':
               loss_train /= accum_iter

            loss_train.backward()
            if args.w_grad_update_method in ['mean', 'sum']:
               if device_idx + 1 == accum_iter:
                  nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                  optimizer.step()
                  optimizer.zero_grad()
                  optimizer_arch.zero_grad()
               else:
                  continue
            else:
               nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
               optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss_train.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)
            print(arch_params)
            print(scalarization)
            #print('HPN grad norm: {}'.format(g_norm))
        #break
        #if step >=10:
        #    break
        #    break
        #break
        # if 'debug' in args.save:
        #    break

    return top1.avg, objs.avg


def infer(model, hpn, hpn_rand, args, epoch, help_loader):
    model.eval()

    circle_points_test = circle_points(args.num_test)
    cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    hw_embeddings, metrics = help_loader.sample_all_devices(mode='all')
    hw_embeddings = [hw_emb.cuda() for hw_emb in hw_embeddings]

    for i, hw_emb in enumerate(hw_embeddings):
        results_test = {"rays":[], "archs":[]}
        results_test_random = {"rays":[], "archs":[]}
        results_test_hpn_random = {"rays":[], "archs":[]}

        metric_name = metrics[i]
        with torch.no_grad():
            for point in circle_points_test:
                scalarization = torch.tensor([point[0], point[1]]).cuda()
                # convert to float
                scalarization = scalarization.float()
                arch_params = hpn(scalarization.unsqueeze(0),
                                  hw_emb.unsqueeze(0))
                model.set_arch_params(arch_params)
                genotype = model.genotype().tostr()
                arch = genotype
                results_test["rays"].append(point)
                results_test["archs"].append(arch)

                # random hypernet
                arch_params = hpn_rand(scalarization.unsqueeze(0),
                                       hw_emb.unsqueeze(0))
                model.set_arch_params(arch_params)
                genotype = model.genotype().tostr()
                arch = genotype
                results_test_hpn_random["rays"].append(point)
                results_test_hpn_random["archs"].append(arch)

                # sample a random architecture (torch.tensor of shape (5, 6))
                arch_params = torch.eye(5)[np.random.choice(5, 6)]
                model.set_arch_params(arch_params)
                genotype = model.genotype().tostr()
                arch = genotype
                results_test_random["rays"]. append(point)
                results_test_random["archs"].append(arch)

        #print(results_test["archs"])
        metrics_dict = analysis(results_test["archs"],
                                results_test_random["archs"],
                                results_test_hpn_random["archs"], model,
                                metric_name, epoch, args.dataset, args,
                                rays=None, #results_test["rays"],
                                help_loader=None)

        # add hypervolume_true, hypervolume_predicted, gd, igd, igd_plus, gd_plus
        #wandb.log(metrics_dict)
        #error_lat = np.array([predicted_errors, predicted_latencies]).T
        #rays = np.array(results_test["rays"])
        #cosine_similarities = cosine_sim(torch.tensor(error_lat), torch.tensor(rays))
        #print("Cosine Similarities: ", torch.mean(cosine_similarities))
        #wandb.log({metric_name+"_cosine_sim": torch.mean(cosine_similarities)})

    return results_test


if __name__ == '__main__':
    main()
