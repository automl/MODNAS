from optimizers.mixop.base_mixop import MixOp

import torch
import torch.nn as nn

class EntangledOp(nn.Module):
    def __init__(self, op, name):
        super(EntangledOp, self).__init__()
        self.op = op
        self.name = name

    def forward(self, x, weights, use_argmax=False):
        #print(self.op)
        return self.op(x, weights, use_argmax=use_argmax)

    def wider(self, C1, C2):
        if self.op!=None:
           self.op.wider(C1, C2)

class EntangleMixOp(MixOp):

    def forward(self, x, weights, ops, add_params=False, combi=False):
        """ Forward pass through the MixedOp

        add_params and combi are ignored and do not have any effect
        """
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights)

        entangled_op_weights = {}
        entangled_ops = {}

        out = 0
        i = 0
        for w, op in zip(weights, ops):
            if isinstance(op, EntangledOp):
                if op.name not in entangled_op_weights:
                    entangled_op_weights[op.name] = [w]
                else:
                    entangled_op_weights[op.name].append(w)

                if (op.name not in entangled_ops) or (entangled_ops[op.name] is None):
                    entangled_ops[op.name] = op if op.op is not None else None
                i = i+1
                continue
            #print(op)
            #print(op(x).shape)
            out = out + w * op(x)
        for op_name in entangled_op_weights.keys():
            w = entangled_op_weights[op_name]
            op = entangled_ops[op_name]
            out = out + op(x, w)
        return out

    def forward_progressive(self,
                            x,
                            weights,
                            ops,
                            add_params=False,
                            combi=False):
        weights = self.preprocess_weights(weights)

        entangled_op_weights = {}
        entangled_ops = {}

        out = 0
        for w, op in zip(weights, ops):
            if isinstance(op, EntangledOp):
                if op.name not in entangled_op_weights:
                    entangled_op_weights[op.name] = [w]
                else:
                    entangled_op_weights[op.name].append(w)

                if (op.name not in entangled_ops) or (entangled_ops[op.name] is None):
                    entangled_ops[op.name] = op if op.op is not None else None
                continue
            if not w == 0:
                #print(op)
                
                out = out + w * op(x)
                #print(out.shape)

        for op_name in entangled_op_weights.keys():
            weights = entangled_op_weights[op_name]
            #assert len(weights) == 2 # Assume only two operations are entangled at once. E.g., a 1x1 and a 3x3 conv. Not, for example, a 1x1, a 3x3 and a 5x5 conv.
            #print(op_name)
            op = entangled_ops[op_name]
            out = out + op(x, weights)

        return out

    def forward_layer(self,
                      x,
                      weights,
                      ops,
                      master_op,
                      add_params=False,
                      combi=False):
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi([weights[0], weights[1]])
        entangled_ops = {}
        entangled_op_weights = {}
        out = 0
        i =  0
        for w, op in zip(weights, ops):
            if isinstance(op, EntangledOp):
                if op.name not in entangled_op_weights:
                    entangled_op_weights[op.name] = [w]
                else:
                    entangled_op_weights[op.name].append(w)
                if (op.name not in entangled_ops) or (entangled_ops[op.name] is None):
                    entangled_ops[op.name] = op if op.op is not None else None
                i = i+1
                continue
            out = out + w * op(x, master_op)

        for op_name in entangled_op_weights.keys():
            weights = entangled_op_weights[op_name]
            #assert len(weights) == 2 # Assume only two operations are entangled at once. E.g., a 1x1 and a 3x3 conv. Not, for example, a 1x1, a 3x3 and a 5x5 conv.
            #print(op_name)
            op = entangled_ops[op_name]
            out = out + op(x, weights)
        return out

    def forward_layer_2_outputs(self,
                                x,
                                weights,
                                ops,
                                master_op,
                                add_params=False):
        raise NotImplementedError

    def forward_layer_2_inputs(self,
                               x1,
                               x2,
                               weights,
                               ops,
                               master_op,
                               add_params=False):
        raise NotImplementedError

    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        raise NotImplementedError

    def forward_swin_attn(self,
                          x,
                          weights,
                          ops,
                          mask,
                          B_,
                          N,
                          add_params=False,
                          combi=False):
        raise NotImplementedError
