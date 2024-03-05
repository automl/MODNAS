# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from search_spaces.hat.fairseq import utils
from search_spaces.hat.fairseq.utils import get_space
from search_spaces.hat.fairseq.criterions import FairseqCriterion, register_criterion
from hypernetworks.models.hpn_hat import convert_arch_params_to_dict
import pickle 

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0]*2))
        self.latency_trajectory = []
        self.ce_trajectory = []
        with open("predictor_data_utils/hat/hat_latecy_stat_dict.pkl", "rb") as f:
             self.latency_dict = pickle.load(f)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, hypernetwork, hw_embed, task, device, scalarization=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if scalarization is None:
           s = self.dirichlet.sample().unsqueeze(0).to(sample['target'].device)
        else:
           s = scalarization.unsqueeze(0).to(sample['target'].device)#.half()
        #hypernetwork = hypernetwork.half()
        arch_params_pred = hypernetwork(s, hw_embed)
        arch_params_dict = convert_arch_params_to_dict(arch_params_pred)
        space = get_space("space0")
        model.set_sample_config(space, arch_params_dict)
        hw_embed = hw_embed.to(sample['target'].device)
        net_output, latency = model(**sample['net_input'],arch_param=arch_params_dict, hw_embed=hw_embed)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        if len(self.ce_trajectory)<1:
           loss_scalarized = s[0][0]*loss + s[0][1]*latency
        else:
            if task == "wmt16_en_de":
                task = "wmt14_en_de"
            latency_normalized = (latency - self.latency_dict[task][device]["min"])/(self.latency_dict[task][device]["max"] - self.latency_dict[task][device]["min"])
            latency_normalized = latency_normalized*(max(self.ce_trajectory) - min(self.ce_trajectory)) + min(self.ce_trajectory)
            loss_scalarized = s[0][0]*loss + s[0][1]*latency_normalized
        if model.training:
           self.ce_trajectory.append(loss.item())
           self.latency_trajectory.append(latency.item())
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'loss_scalarized': utils.item(loss_scalarized.data) if reduce else loss_scalarized.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output,  loss, latency

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
