from optimizers.sampler.base_sampler import Sampler
import torch
import torch.nn.functional as F
from reinmax import reinmax

class ReinmaxSampler(Sampler):

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
        #print(alpha.shape)

        alpha_sampled = reinmax(alpha, 1)[0]
        #print(alpha_sampled.shape)
        return alpha_sampled

    def before_epoch(self):
        if self.tau_max is None:
            raise Exception('tau_max has to be set in GDASSampler')

        self.tau_curr = self.tau_max - (
            self.tau_max - self.tau_min) * self.epoch / (self.total_epochs - 1)
        self.epoch += 1
