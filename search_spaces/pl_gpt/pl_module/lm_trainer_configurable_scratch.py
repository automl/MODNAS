import time
import warnings
import numpy as np
from collections import defaultdict
import pytorch_lightning as pl
import torch
import inspect
import random
from search_spaces.gpt.model import GPT
from search_spaces.gpt.utils import *
from search_spaces.pl_gpt.utils import instantiate, get_class
from search_spaces.pl_gpt.utils.group_parameters import group_parameters_for_optimizer
from search_spaces.pl_gpt.utils.optim.lr_schedule import get_learning_rate_schedule
from hypernetworks.hpn_gpt import MetaHyperNetwork
from predictors.gpt.hw_loader import search_spaces, HWDataset
from optimizers.mgd import MinNormSolver as MGD
class LanguageModelTrainer(pl.LightningModule):
    """
    PTL wrapper class for model training
    """

    def __init__(
            self,
            cfg_train,
            cfg_model,
            py_logger,
            val_sets_name,
            ignore_index,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.cfg_train = cfg_train
        self.cfg_model = cfg_model

        self.val_sets_name = val_sets_name
        self.ignore_index = ignore_index
        self.py_logger = py_logger

        #self.model = GPT(cfg_model)
        
        self.loss_train = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean', label_smoothing=0.0)
        #self.loss_train = FlashCELoss(ignore_index=self.ignore_index, reduction='mean', label_smoothing=0.0,
        #                              inplace_backward=True)

        self.intern_log = []
        self.log_lists = defaultdict(list)

        self.validation_step_outputs = defaultdict(list)
        # build choices dict
        self.choices_dict = search_spaces["s"]
        self.model = GPT(cfg_model, choices_dict=self.choices_dict, ignore_index=ignore_index)
        # set requires grad to False for all model params
        #for p in self.model.parameters():
        #    p.requires_grad = False
        checkpoint_path = cfg_model.checkpoint_path
        state_dict = {}
        if checkpoint_path is not None:
            state_dict_loaded = torch.load(checkpoint_path, map_location='cpu')['state_dict']
            for k, v in state_dict_loaded.items():
                state_dict[k.replace("model.", "")] = v
        #self.model.load_state_dict(state_dict, strict=False)
        self.hpn = MetaHyperNetwork(self.choices_dict)
        #self.hpn.load_state_dict(torch.load("pretrained_hpns/mhn_gpt.pth"))
        self.hwdset = HWDataset()
        self.devices_all = self.hwdset.gpus
        self.devices_train = self.hwdset.gpus[0:4]
        self.devices_val = self.hwdset.gpus[4:8]
        self.scheme = cfg_model.sampling_scheme
        self.train_strategy = cfg_model.train_strategy
        self.sandwhich_random = cfg_model.sandwhich_random
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0, 1.0]))
        self.init_max_min()
        #self.automatic_optimization = False

    def init_max_min(self):
        self.config_max = sample_config_max(self.choices_dict, layer_sampling_scheme=self.scheme)
        self.config_min = sample_config_min(self.choices_dict, layer_sampling_scheme=self.scheme)
        self.config_mid =  sample_config_mid(self.choices_dict, layer_sampling_scheme=self.scheme)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        #seed = hash((self.global_step,self.current_epoch,self.local_rank))

        #sampled_config = self.get_arch_sampled(seed)
        #sample_intermediate_size = [sampled_config["sample_mlp_ratio"][i]*sampled_config["sample_embed_dim"] for i in range(len(sampled_config["sample_mlp_ratio"]))]
        #self.model.set_sample_config(sampled_config["sample_embed_dim"], sample_intermediate_size , sampled_config["sample_n_head"], sampled_config["sample_n_layer"], sampled_config["sample_bias"], sampled_config["sample_layer_indices"])
        #print(sample_config)
        device = self.devices_train[self.local_rank]
        batch_hw = self.hwdset.sample_batch(device, 1, "train")[2].float().cuda()
        scalarization = self.dirichlet.sample().cuda().unsqueeze(0)
        arch_params = self.hpn(scalarization, batch_hw)
        loss, energy, logits = self.model(idx=batch['src_seq'], arch_params=arch_params, device_name=device,labels=batch['trg_seq'].view(-1),hw_embed=batch_hw)
        print(loss,energy)
        labels = batch['trg_seq'].view(-1)
        #loss = self.loss_train(logits.view(-1, logits.size(-1)), labels)
        loss_final = scalarization[0][0]*loss + scalarization[0][1]*energy
          #loss = loss#/n
        loss_value = loss_final.detach()#*n
        self.log(
            f"train/loss",
            loss_value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss_final}
    
    def all_reduce(self, metrics):
        metrics_reduce = {}
        for key, value in metrics.items():
            mean_value = value.mean()
            metrics_reduce[key] = mean_value.item()
        
        
        return metrics_reduce
    
    def on_before_optimizer_step(self, optimizer):
        list_of_grads = {}
        for i,param in enumerate(self.hpn.parameters()):
            list_of_grads[i] = [torch.zeros_like(param.grad) for _ in range(len(self.devices_train))]

        list_of_grads_mgd = []
        # get the gradients of the hypernetwork for al ranks
        # gather all grads from different local ranks
        for i,param in enumerate(self.hpn.parameters()):
            torch.distributed.all_gather(list_of_grads[i], param.grad)
            #print(param.grad)
            #break
        #print(list_of_grads)
        for i in range(len(list_of_grads[0])):
            param_grad_list = []
            for k in list_of_grads.keys():
                param_grad_list.append(list_of_grads[k][i])
            gn = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in param_grad_list]))
            for gr_i in range(len(param_grad_list)):
                param_grad_list[gr_i] /= gn
            list_of_grads_mgd.append(param_grad_list)
        # solve the min norm problem
        #print(list_of_grads_mgd)
        gammas, _ = MGD.find_min_norm_element_FW(list_of_grads_mgd)
        # sum the scaled gradients
        mgrad = [torch.zeros_like(param) for param in list_of_grads_mgd[0]]
        for gamma, gr_i in zip(gammas, list_of_grads_mgd):
            gr_i = [gamma*gr for gr in gr_i]
            mgrad = list(map(lambda x, y: x + y, gr_i, mgrad))
        # set the hypernetwork gradients to the min norm gradients
        for param, mg in zip(self.hpn.parameters(), mgrad):
            param.grad = torch.zeros_like(param)
            param.grad.data.copy_(mg)
        #for param in self.hpn.parameters():
        #    print(param.grad)
        #    break

    def validation_step(self, batch, batch_idx , dataloader_idx=0):
        device = self.devices_val[self.local_rank]
        batch_hw = self.hwdset.sample_batch(device, 1, "train")[2].float().cuda()
        scalarization = self.dirichlet.sample().cuda().unsqueeze(0)
        arch_params = self.hpn(scalarization, batch_hw)

        with torch.no_grad():
            loss, energy, logits = self.model(idx=batch['src_seq'], arch_params=arch_params, device_name=device,labels=batch['trg_seq'].view(-1),hw_embed=batch_hw)

        labels = batch['trg_seq'].view(-1)
        #loss = self.loss_train(logits.view(-1, logits.size(-1)), labels)
        loss = scalarization[0][0]*loss + scalarization[0][1]*energy
        if dataloader_idx == 0:
            self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return_dict = {"loss": loss,
                       "batch_size": torch.FloatTensor([batch['trg_len'].shape[0]]),
                       "batch_length": torch.mean(batch['trg_len'].detach().float()),
                       "num_loss_tokens": torch.sum(batch['trg_len'])
                       }

        count = torch.sum(batch['trg_len'], dtype=torch.float)
        log_probs = loss * count
        preds = logits.argmax(dim=-1).view(-1)
        target = batch['trg_seq'].view(-1)
        idx = target != self.ignore_index
        accuracy = torch.sum(preds[idx] == target[idx])

        return_dict.update({"accuracy": accuracy, "log_probs": log_probs, "count": count})

        self.validation_step_outputs[dataloader_idx].append(return_dict)

        return return_dict

    def on_validation_epoch_end(self):

        values = ['log_probs', 'accuracy', 'count']

        assert len(self.val_sets_name) == len(self.validation_step_outputs)

        for dataset_idx, dataset_name in enumerate(self.val_sets_name):

            output = self.validation_step_outputs[dataset_idx]
            summed_values = {k: 0 for k in values}
            for out_dict in output:
                for key in values:
                    summed_values[key] += out_dict[key]

            ppl = torch.exp(summed_values['log_probs'] / summed_values['count'])
            accuracy = summed_values['accuracy'] / summed_values['count']
            metrics = {"ppl": ppl, "acc": accuracy}
            #print(metrics)
            for name, value in metrics.items():
                self.log(f"val/{dataset_name}/{name}", value,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, )
            # g
        

        # check rank is 0
        #if self.global_rank == 0:
        #if self.local_rank == 0:
        self.current_metrics = self.all_reduce(self.all_gather(metrics))
        if self.local_rank == 0:
            self.py_logger.info(f"Validation metrics: {self.current_metrics}")
        #else:
        #    self.current_metrics = None
        self.validation_step_outputs.clear()


    def configure_optimizers(self):

        if 'optimizer_param_grouping' in self.cfg_train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg_train.optimizer,
                                                        **self.cfg_train.optimizer_param_grouping)
            parameters.append({"params": self.hpn.parameters(), "lr": 1e-5})
        else:
            #parameters = self.model.parameters()
            parameters = [{"params": self.model.parameters()},{"params":self.hpn.parameters(), "lr":1e-5}]

        optimizer = instantiate(self.cfg_train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            self.py_logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg_train:
            return optimizer
        else:
            lr_lambda = get_learning_rate_schedule(self.cfg_train.scheduler)

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg_train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg_train.get('scheduler_monitor', 'val/loss')}
       

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
        if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()

    #def on_train_epoch_start(self):
    #    random.seed(self.current_epoch)
