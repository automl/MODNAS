# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import torch
from search_spaces.hat.fairseq import utils

from . import FairseqCriterion, register_criterion
from search_spaces.hat.fairseq.utils import get_space
from hypernetworks.models.hpn_hat import convert_arch_params_to_dict
import pickle
@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0]*2))
        self.latency_trajectory = []
        self.ce_trajectory = []
        with open("predictor_data_utils/hat/hat_latecy_stat_dict.pkl", "rb") as f:
             self.latency_dict = pickle.load(f)

    def forward(self, model, sample,  hypernetwork, hw_embed, task, device, scalarization=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # sample scalarization
        if scalarization is None:
           s = self.dirichlet.sample().unsqueeze(0).to(sample['target'].device)
        else:
           s = scalarization.unsqueeze(0).to(sample['target'].device)
        arch_params_pred = hypernetwork(s, hw_embed)
        arch_params_dict = convert_arch_params_to_dict(arch_params_pred)
        space = get_space("space0")
        model.set_sample_config(space, arch_params_dict)
        net_output, latency = model(**sample['net_input'], arch_param = arch_params_dict, hw_embed=hw_embed)
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        if len(self.ce_trajectory)<1:
           loss_scalarized = s[0][0]*loss + s[0][1]*latency

        else:
            latency_nromalized = (latency - self.latency_dict[task][device]["min"])/(self.latency_dict[task][device]["max"] - self.latency_dict[task][device]["min"])
            latency_normalized = latency_nromalized*(max(self.ce_trajectory) - min(self.ce_trajectory)) + min(self.ce_trajectory)
        if model.training:
           self.ce_trajectory.append(loss.item())
           self.latency_trajectory.append(latency.item())
        
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss_scalarized': utils.item(loss_scalarized.data) if reduce else loss_scalarized.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss_scalarized, sample_size, logging_output, loss, latency

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
