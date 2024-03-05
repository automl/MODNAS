import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from collections.abc import MutableMapping
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
from nas_201_api import NASBench201API as API

from search_spaces.abstract_model import SearchNetworkBase
from search_spaces.nb201.operations import OPS, ReLUConvBNSubSample, \
    ResNetBasicblock, ReLUConvBNMixture, \
    NAS_BENCH_201
from search_spaces.nb201.genotypes import Structure
from search_spaces.nb201.hw_nas_bench_api import HWNASBenchAPI as HWAPI
from optimizers.optim_factory import get_mixop, get_sampler
from utils import add_global_node
from predictors.nb201.models.predictors import PredictorNB201, GCN, \
        GCNHardware, MetaLearner


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

class NB201MixedOpWrapper(nn.Module):

    def __init__(self, mixop, primitives, entangle_weights=True):
        super(NB201MixedOpWrapper, self).__init__()
        self._ops = torch.nn.ModuleDict()
        self.mixop = mixop
        self.entangle_weights = entangle_weights

        if entangle_weights == True:
            self._init_entangled_ops(primitives)
        else:
            self._init_ops(primitives)

    def _init_entangled_ops(self, primitives):
        for primitive, op in primitives:
            if primitive == 'nor_conv_1x1':
                self._ops[primitive] = ReLUConvBNSubSample(
                    self._ops['nor_conv_3x3'], 1)
            else:
                self._ops[primitive] = op

    def _init_ops(self, primitives):
        for primitive, op in primitives:
            self._ops[primitive] = op

    def forward(self, x, weights):
        return self.mixop.forward(x, weights, list(self._ops.values()))

    def __repr__(self):
        s = f'Operations {list(self._ops.keys())}'
        if self.entangle_weights == True:
            s += ' with entangled weights'

        return s


class NAS201SearchCell(nn.Module):

    def __init__(
        self,
        optimizer_type,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
        entangle_weights=True,
    ):
        super(NAS201SearchCell, self).__init__()
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        self.mixop = get_mixop(optimizer_type)

        # Stats variables for tracking gradient contribution
        self.stats_total_grad_inputs = {}
        self.stats_backward_steps ={}
        self.stats_grad_norms = {}
        self.stats_last_grad_inputs = {}
        self.is_architect_step = False

        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    primitives = [(op_name, OPS[op_name](C_in, C_out, stride,
                                                         affine,
                                                         track_running_stats))
                                  for op_name in op_names]
                else:
                    primitives = [(op_name, OPS[op_name](C_in, C_out, 1,
                                                         affine,
                                                         track_running_stats))
                                  for op_name in op_names]
                self.edges[node_str] = NB201MixedOpWrapper(self.mixop, primitives, entangle_weights=entangle_weights)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__)
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class NASBench201SearchSpace(SearchNetworkBase):

    def __init__(self,
                 C,
                 N,
                 max_nodes,
                 num_classes,
                 criterion=torch.nn.CrossEntropyLoss().to(DEVICE),
                 optimizer_type='gdas',
                 search_space=NAS_BENCH_201,
                 affine=False,
                 track_running_stats=False,
                 reg_type='l2',
                 reg_scale=1e-3,
                 path_to_benchmark='datasets/NAS-Bench-201-v1_0-e61699.pth',
                 path_to_hw_benchmark='datasets/HW-NAS-Bench-v1_0.pickle',
                 entangle_weights=False,
                 initialize_api = True,
                 track_gradients=False,
                 metric = "fpga_latency",
                 datset = "cifar10",
                 hw_embed_on=False,
                 hw_embed_dim=10,
                 layer_size=100,
                 load_path=None,
                 latency_norm_scheme="batch_stats"):
        super(NASBench201SearchSpace, self).__init__()
        self.optimizer_type = optimizer_type
        self.sampler = get_sampler(optimizer_type)
        self.dataset = datset
        self.op_names = deepcopy(list(reversed(search_space)))

        self._C = C
        self._N = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))
        self._criterion = criterion
        self.num_classes = num_classes
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.reg_type = reg_type
        self.reg_scale = reg_scale
        self.latency_norm_scheme = latency_norm_scheme
        self.entangle_weights = entangle_weights
        self.track_gradients = track_gradients
        self.metric = metric
        self.hw_embed_on = hw_embed_on
        self.hw_embed_dim = hw_embed_dim
        self.layer_size = layer_size
        self.load_path = load_path

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4
                                                            ] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
                zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(
                    self.optimizer_type,
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    self.op_names,
                    affine,
                    track_running_stats,
                    entangle_weights=entangle_weights
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (num_edge == cell.num_edges and edge2index
                            == cell.edge2index), "invalid {:} vs. {:}.".format(
                                num_edge, cell.num_edges)

            self.cells.append(cell)
            C_prev = cell.out_dim
        self.num_edge = num_edge
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        #self.predictor = GCNHardware(8,True,10,100).cuda()
        #self.predictor.load_state_dict(torch.load(self.get_predictor_path(metric)))
        self.predictor = MetaLearner('nasbench201',
                                     hw_embed_on=hw_embed_on,
                                     hw_embed_dim=hw_embed_dim,
                                     layer_size=layer_size)
        if load_path is not None:
            if "help_max_corr.pt" in load_path:
                self.predictor.load_state_dict(torch.load(load_path)['model'])
            else:
                self.predictor.load_state_dict(torch.load(load_path))
            print('Loaded HELP MetaLearner predictor weights...')
        # turn gradients off for predictor
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.type = "base"
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev),
                                     nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.path_to_benchmark = path_to_benchmark
        if initialize_api:
            self.api = API(path_to_benchmark, verbose=False)
            self.hw_api = HWAPI(path_to_hw_benchmark, search_space="nasbench201")
            print('Loaded NB201 and HW-NAS-Bench datasets...')
        #self.api = None
        self._initialize_alphas()
        self.ce_loss = []
        self.latencies = []

        #self._initialize_anchors()

    def get_predictor_path(self, metric="fpga_latency"):
        return "predictor_data_utils/nb201/predictor_meta_learned.pth"

    def _initialize_anchors(self):
        self.anchor = Dirichlet(
            torch.ones_like(self._arch_parameters[0]).to(DEVICE))

    def process_for_predictor(self, arch_params):
        #print(arch_params)
        architects = torch.load("predictor_data_utils/nb201/architecture.pt")
        str_to_idx = torch.load("predictor_data_utils/nb201//str_arch2idx.pt")
        self.set_arch_params(arch_params)
        genotype, arch_params_ids = self.genotype_predictor()
        arch_str = genotype.tostr()
        #print(arch_str)
        arch = str_to_idx[arch_str]
        arch_info = architects[arch]
        processed_operations = add_global_node(arch_info["operation"],ifAdj=False)#.unsqueeze(0)
        processed_adjacency = add_global_node(arch_info["adjacency_matrix"],ifAdj=True)#.unsqueeze(0)
        processed_operations[1:processed_operations.shape[0]-2,1:6] = arch_params[:,arch_params_ids]
        #print(processed_operations)
        #print(add_global_node(arch_info["operation"],ifAdj=False))
        assert torch.allclose(processed_operations, add_global_node(arch_info["operation"],ifAdj=False), atol=1e-05), "processed_operations not equal to original"

        return (processed_operations.unsqueeze(0).cuda(), processed_adjacency.unsqueeze(0).cuda()), arch_str
    
    def get_predictor_gt_stats(self, metric_name = "fpga_latency"):
        with open("predictor_data_utils/nb201/all_stats.pkl","rb") as f:
            stats = pickle.load(f)
        return stats[metric_name]

    def _initialize_alphas(self):
        self.arch_parameter = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(self.op_names)))
        self._arch_parameters = [self.arch_parameter]

    def set_arch_params(self, arch_param):
        self.arch_parameter = nn.Parameter(arch_param)
        self.arch_parameter.data = arch_param.data
        
        self._arch_parameters = [self.arch_parameter]

    def _get_kl_reg(self):
        cons = (F.elu(self._arch_parameters[0]) + 1)
        q = Dirichlet(cons)
        p = self.anchor
        kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
        return kl_reg

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(self._arch_parameters[0], dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(i, len(self.cells),
                                                       cell.extra_repr())
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_N}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__)

    def genotype_predictor(self):
        genotypes = []
        ops_help = ["avg_pool_3x3","nor_conv_1x1","nor_conv_3x3","none","skip_connect"]
        # permute alphas from search space in this order 
        map = {"avg_pool_3x3": 0, "nor_conv_1x1": 2, "nor_conv_3x3": 1, "none": 4, "skip_connect": 3}
        indices_permute = [map[op] for op in ops_help]
        arch_param = self._arch_parameters[0]
        alphas = torch.nn.functional.softmax(arch_param, dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = alphas[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes),indices_permute
    
    def genotype(self):
        genotypes = []
        arch_param = self._arch_parameters[0]
        alphas = torch.nn.functional.softmax(arch_param, dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = alphas[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def query(self):
        result = self.api.query_by_arch(self.genotype(), '200')
        return result

    def query_hw(self, dataset=None):
        if dataset is not None:
            assert(dataset in ["cifar10", "cifar100", "ImageNet16-120"])

        result = dict()
        for d in ["cifar10", "cifar100", "ImageNet16-120"]:
            HW_metrics = self.hw_api.query_by_index(
                self.api.query_index_by_arch(self.genotype()),
                d
            )
            result.update({d: HW_metrics})

        return result if dataset is None else result[dataset]

    def reset_batch_stats(self):
        self.ce_loss = []

    def forward(self, inputs, labels, alphas=None, hw_embed=None, hw_metric="",
                discretize=False, fix_norm=False):
        if discretize:
            argmax_alphas = torch.argmax(alphas, dim=-1)
            alphas = torch.zeros_like(alphas)
            alphas[torch.arange(alphas.shape[0]), argmax_alphas] = 1.0
        else:
          alphas = self.sampler.sample_step([alphas])[0]
        feature = self.stem(inputs)
        #print(alphas)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, NAS201SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        latency = self.predictor(self.process_for_predictor(alphas)[0], hw_embed)

        self.set_arch_params(alphas)
        arch = self.genotype()
        arch = arch.tostr()
        #latency = torch.nn.functional.sigmoid(latency)
        ce_loss = self._criterion(logits, labels)
        if len(self.ce_loss) > 1 :
            # max-min normalization
            if fix_norm:
                ce_loss_normalized = ce_loss
            else:
                ce_loss_normalized = (ce_loss - min(self.ce_loss)) / (max(self.ce_loss) - min(self.ce_loss))
            if self.latency_norm_scheme == "batch_stats":
                 min_lat, max_lat = max(self.latencies), min(self.latencies)
            elif self.latency_norm_scheme == "predictor_stats":       
                lat_stats = self.get_predictor_gt_stats(metric_name=hw_metric)
                min_lat = lat_stats["min"]
                max_lat = lat_stats["max"]
            latency_normalized  = (latency - min_lat) / (max_lat - min_lat)
            if fix_norm:
                latency_normalized *= (max(self.ce_loss) - min(self.ce_loss))
                latency_normalized += min(self.ce_loss)
        else:
            ce_loss_normalized = ce_loss
            latency_normalized = latency
        if not discretize:
            self.ce_loss.append(ce_loss.item())
            self.latencies.append(latency.item())
        return logits, ce_loss_normalized, latency_normalized, arch

    def _loss(self, input, target):
        _, logits = self(input)
        loss = self._criterion(logits, target)
        return loss, logits

    def _get_kl_reg(self):
        cons = (F.elu(self._arch_parameters[0]) + 1)
        q = Dirichlet(cons)
        p = self.anchor
        kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
        return kl_reg


    def new(self):
        model_new = NASBench201SearchSpace(
            self.optimizer_type,
            C=self._C,
            N=self._N,
            max_nodes=self.max_nodes,
            num_classes=self.num_classes,
            search_space=list(reversed(self.op_names)),
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            criterion=self._criterion,
            reg_type=self.reg_type,
            reg_scale=self.reg_scale,
            path_to_benchmark=self.path_to_benchmark,
            entangle_weights = self.entangle_weights,
            initialize_api=False,
            hw_embed_on=self.hw_embed_on,
            hw_embed_dim=self.hw_embed_dim,
            layer_size=self.layer_size,
            load_path=self.load_path
        ).to(DEVICE)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model_new

    def get_saved_stats(self):
        stats = {}
        grad_norms_flat = {}

        def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
            items = []
            for k, v in d.items():
                new_key = str(parent_key) + sep + str(k) if parent_key else str(k)
                if isinstance(v, MutableMapping):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for idx, cell in enumerate(self.cells):

            if not isinstance(cell, NAS201SearchCell):
                continue

            stats[idx] = {}
            stats[idx]['avg_grad_inputs'] = {}

            for edge_op_name in cell.stats_total_grad_inputs.keys():
                stats[idx]['avg_grad_inputs'][edge_op_name] = cell.stats_total_grad_inputs[edge_op_name]/cell.stats_backward_steps[edge_op_name]

            stats[idx]['grad_norms'] = cell.stats_grad_norms
            stats[idx]['last_grad_inputs'] = cell.stats_last_grad_inputs
            grad_norms_flat[idx] = flatten_dict(cell.stats_grad_norms)

        stats['grad_norms_flat'] = flatten_dict(grad_norms_flat)
        return stats

    @property
    def is_architect_step(self):
        is_arch_steps = []
        for cell in self.cells:
            if isinstance(cell, NAS201SearchCell):
                is_arch_steps.append(cell.is_architect_step)

        assert (all(is_arch_steps) or all([not a for a in is_arch_steps]))
        return is_arch_steps[0]

    @is_architect_step.setter
    def is_architect_step(self, value):
        for cell in self.cells:
            if isinstance(cell, NAS201SearchCell):
                cell.is_architect_step = value

    def make_alphas_for_genotype(self, genotype):
        ops = [i for g in genotype.tolist("")[0] for i in g]
        alphas = torch.zeros_like(self.arch_parameters()[0])

        for idx, op in enumerate(ops):
            alphas[idx][self.op_names.index(op[0])] = 1.0

        return alphas

    def set_alphas_for_genotype(self, genotype):
        alphas = self.make_alphas_for_genotype(genotype)
        self.arch_parameters()[0].data = alphas.data

    def sample(self):
        self._initialize_alphas()
        genotype = self.genotype()
        self.set_alphas_for_genotype(genotype)
        return genotype

    def mutate(self, p=0.5):
        new_alphas = torch.zeros_like(self.arch_parameters()[0])

        for idx, row in enumerate(self.arch_parameters()[0]):
            if np.random.rand() < p:
                new_op = np.random.randint(0, new_alphas.shape[1])
                new_alphas[idx][new_op] = 1.0
            else:
                new_alphas[idx] = row.detach()

        self.arch_parameters()[0].data = new_alphas.data
        return self.genotype()

    def crossover(self, other_genotype, p=0.5):
        new_alphas = torch.zeros_like(self.arch_parameters()[0])
        other_alphas = self.make_alphas_for_genotype(other_genotype)

        for idx, row in enumerate(self.arch_parameters()[0]):
            if np.random.rand() < p:
                new_alphas[idx] = row.detach()
            else:
                new_alphas[idx] = other_alphas[idx].detach()

        self.arch_parameters()[0].data = new_alphas.data
        return self.genotype()

