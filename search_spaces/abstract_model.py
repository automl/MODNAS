import torch

from abc import ABC, abstractmethod


class NetworkBase(ABC, torch.nn.Module):

    def get_weights(self):  # TODO: find a cleaner way to do this
        li = []
        for n, p in self.named_parameters():
            if ("alpha" not in n) and ("arch" not in n) and p!=None and ("beta" not in n):
                li.append(p)
        return li

    def get_named_weights(self):  # TODO: find a cleaner way to do this
        li = {}
        for n, p in self.named_parameters():
            if ("alpha" not in n) and ("arch" not in n) and ("beta" not in n):
                li[n] = p
        for k in li.keys():
            yield (k, li[k])

    def arch_parameters(self):
        return self._arch_parameters

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _initialize_alphas(self):
        pass


class SearchNetworkBase(NetworkBase):

    @abstractmethod
    def show_alphas(self):
        pass

    @abstractmethod
    def new(self):
        pass

    @abstractmethod
    def _loss(self, input, target):
        pass

    def get_saved_stats(self):
        return {}

    @property
    def is_architect_step(self):
        pass

    @is_architect_step.setter
    def is_architect_step(self, value):
        pass
