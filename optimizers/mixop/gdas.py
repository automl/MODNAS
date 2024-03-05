from optimizers.mixop.base_mixop import MixOp
import torch


class GDASMixOp(MixOp):

    def preprocess_weights(self, weights):
        return weights

    def preprocess_combi(self, weights):
        out = 0
        if len(weights) == 2:
            out = weights[0].reshape(weights[0].shape[0], 1) @ weights[1].reshape(1, weights[1].shape[0])
            out = out.flatten()
        elif len(weights) == 3:
            out = weights[0].reshape(weights[0].shape[0], 1) @ weights[1].reshape(1, weights[1].shape[0])
            out = out.flatten()
            out = out.reshape(out.shape[0], 1) @ weights[2].reshape(1, weights[2].shape[0])
            out = out.flatten()
        return out

    def forward(self, x, weights, ops, add_params=False, combi=False):
        ops = list(ops)
        if combi == True:
            weights = self.preprocess_combi(weights)


        argmax = torch.argmax(weights)
        out = sum(weights[i] * op(x) if i == argmax else weights[i]
                  for i, op in enumerate(ops))
        params = 0
        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()

            return out, params
        return out

    def forward_progressive(self,
                            x,
                            weights,
                            ops,
                            add_params=False,
                            combi=False):
        ops = list(ops)
        if combi == True:
            weights = self.preprocess_combi(weights)
        argmax = torch.argmax(weights)
        out = sum(weights[i] * op(x) if i == argmax else weights[i]
                  for i, op in enumerate(ops))
        params = 0
        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out

    def forward_layer(self,
                      x,
                      weights,
                      ops,
                      master_op,
                      add_params=False,
                      combi=False):
        if combi == True:
            weights = self.preprocess_combi(weights)
        argmax = torch.argmax(weights)
        out = sum(weights[i] * op(x, master_op) if i == argmax else weights[i]
                  for i, op in enumerate(ops))

        params = 0
        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out

    def forward_layer_2_outputs(self, x, weights, ops, master_op):
        out1 = 0
        out2 = 0
        argmax = torch.argmax(weights)
        for i, op in enumerate(ops):
            if i == argmax:
                x1, x2 = op(x, master_op)
                out1 = out1 + weights[i] * x1
                out2 = out2 + weights[i] * x2
        return out1, out2

    def forward_layer_2_inputs(self, x1, x2, weights, ops, master_op):
        out = 0
        argmax = torch.argmax(weights)
        for i, op in enumerate(ops):
            if i == argmax:
                out = out + weights[i] * op(x1, x2, master_op)
            else:
                out = out + weights[i]
        return out

    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        i = torch.argmax(weights)
        out = weights[i] * x_list[i]
        params = 0
        if add_params == True:
            for w, param in zip(weights, params_list):
                params = params + w * param
            return out, params
        else:
            return out

    def forward_swin_attn(self,
                          x,
                          weights,
                          ops,
                          mask,
                          B_,
                          N,
                          add_params=False,
                          combi=False):
        if combi == True:
            weights = self.preprocess_combi(weights)
        i = torch.argmax(weights)
        out = weights[i] * ops[i](x, mask, B_, N)
        params = 0
        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out