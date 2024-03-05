from abc import ABC, abstractmethod


class Sampler(ABC):

    @abstractmethod
    def sample_epoch(self, alphas_list):
        pass

    @abstractmethod
    def sample_step(self, alphas_list):
        pass

    @abstractmethod
    def sample(self, alpha):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def set_taus(self, tau_min, tau_max):
        pass

    def set_total_epochs(self, total_epochs):
        pass
