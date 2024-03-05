from optimizers.sampler.base_sampler import Sampler
import torch
import torch.nn.functional as F


class GDASSampler(Sampler):

    def __init__(self):
        self.tau_min = None
        self.tau_max = None
        self.total_epochs = None
        self.tau_curr = None
        self.epoch = 0

    def set_taus(self, tau_min, tau_max):
        self.tau_min = torch.Tensor([tau_min])
        self.tau_max = torch.Tensor([tau_max])

    def set_total_epochs(self, total_epochs):
        self.total_epochs = total_epochs

    def sample_epoch(self, alphas_list, sample_subset):
        pass

    def sample_step(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha))
        return sampled_alphas_list

    def sample(self, alpha):
        tau = self.tau_curr.to(alpha.device)
        while True:
            gumbels = -torch.empty_like(
                alpha, device=alpha.device).exponential_().log()
            logits = (alpha.log_softmax(dim=-1) + gumbels) / tau[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits, device=alpha.device).scatter_(
                -1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if ((torch.isinf(gumbels).any()) or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())):
                continue
            else:
                break
        weights = hardwts
        return weights

    def before_epoch(self):
        if self.tau_max is None:
            raise Exception('tau_max has to be set in GDASSampler')

        self.tau_curr = self.tau_max - (
            self.tau_max - self.tau_min) * self.epoch / (self.total_epochs - 1)
        self.epoch += 1
