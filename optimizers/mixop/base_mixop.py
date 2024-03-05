from abc import ABC, abstractmethod


class MixOp(ABC):

    @abstractmethod
    def preprocess_weights(self):
        pass

    @abstractmethod
    def preprocess_combi(self):
        pass

    @abstractmethod
    def forward(self, x, alpha, ops):
        pass

    @abstractmethod
    def forward_layer(self,
                      x,
                      weights,
                      ops,
                      base_op,
                      add_params=False,
                      combi=False):
        pass

    @abstractmethod
    def forward_layer_2_outputs(self,
                                x,
                                weights,
                                ops,
                                base_op,
                                add_params=False):
        pass

    @abstractmethod
    def forward_layer_2_inputs(self,
                               x1,
                               x2,
                               weights,
                               ops,
                               base_op,
                               add_params=False):
        pass

    @abstractmethod
    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        pass

    @abstractmethod
    def forward_swin_attn(self,
                          x,
                          weights,
                          ops,
                          mask,
                          B_,
                          N,
                          add_params=False,
                          combi=False):
        pass
