import torch.nn as nn
import torch
import numpy as np

sin_softmax = lambda x, dim: torch.softmax(torch.sin(x), dim=dim)

class HyperNetwork(nn.Module):
    def __init__(self, num_intervals=101, *args, **kwargs):
        super(HyperNetwork, self).__init__()
        self.embedding = nn.Embedding(num_intervals, 30)
        self.num_intervals = num_intervals

    def forward(self, x):
        id = (x[0][0] * (self.num_intervals-1)).round().int().unsqueeze(0)
        x = self.embedding(torch.tensor([id]).to(x.device))
        return x.reshape(6,5)


class MetaHyperNetwork(nn.Module):
    def __init__(self, num_random_devices=50, hw_embed_dim=10,
                 hpn_type=HyperNetwork, num_intervals=101):
        super(MetaHyperNetwork, self).__init__()
        self.hardware_embedding = nn.Embedding(num_random_devices, hw_embed_dim)
        self.hpn_factory = [hpn_type(num_intervals=num_intervals) for i in range(num_random_devices)]
        self.hpn_factory = nn.ModuleList(self.hpn_factory)
        self.hw_embed_dim = hw_embed_dim

    def forward(self, x, hw):
        # compute similarity between hw and all random devices
        similarity = torch.squeeze(
            torch.matmul(
                hw,
                self.hardware_embedding.weight.transpose(0,1)
            )
        ) / torch.sqrt(torch.tensor(self.hw_embed_dim * 1.0))
        
        similarity = sin_softmax(similarity, dim=-1)
        out = torch.zeros(6, 5).to(x.device)
        for i in range(similarity.shape[0]):
            hpn_out = self.hpn_factory[i](x)
            out+= similarity[i]*hpn_out
        return out.reshape(6,5)


if __name__=="__main__":
    for i in ['MetaHyperNetwork']:
        for j in ['HyperNetwork']:
                hpn = eval(i)(50, 10, eval(j))
                x = torch.tensor([[0.2,0.8]])

                for _ in range(1):
                    hw = torch.nn.functional.softmax(torch.randn(1, 10),dim=-1)
                    hw[hw < 0] = -hw[hw < 0]
                    out = hpn(x, hw)
                    print('\n ###############################################')
                    print(i, j)
                    print(out)
                    print(torch.softmax(out, dim=1))