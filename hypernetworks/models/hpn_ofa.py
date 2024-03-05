import torch.nn as nn
import torch
sin_softmax = lambda x, dim: torch.softmax(torch.sin(x), dim=dim)

def convert_to_dict(x):
    out = {}
    out["ks"] = x[0]
    out["e"] = x[1]
    out["d"] = x[2]
    out["r"] = x[3]
    return out

class HyperNetwork(nn.Module):
    def __init__(self, block_group_info_len, max_depth, kernel_list_len, expand_ratio_list_len, depth_list_len, resolution_len, use_zero_init=False, use_softmax=False):
        super(HyperNetwork, self).__init__()
        self.embedding_kernel = nn.Embedding(101, block_group_info_len*max_depth*kernel_list_len)
        self.embedding_expand_ratio = nn.Embedding(101, block_group_info_len*max_depth*expand_ratio_list_len)
        self.embedding_depth = nn.Embedding(101, block_group_info_len*depth_list_len)
        self.embedding_resolution = nn.Embedding(101,resolution_len)
        self.block_group_info_len = block_group_info_len
        self.max_depth = max_depth
        self.kernel_list_len = kernel_list_len
        self.expand_ratio_list_len = expand_ratio_list_len
        self.depth_list_len = depth_list_len
        self.resolution_len = resolution_len

    def forward(self, x):
        id = int(x[0][0]*100)
        kernel = self.embedding_kernel(torch.tensor([id]).to(x.device))
        expand_ratio = self.embedding_expand_ratio(torch.tensor([id]).to(x.device))
        depth = self.embedding_depth(torch.tensor([id]).to(x.device))
        resolution = self.embedding_resolution(torch.tensor([id]).to(x.device))
        return kernel.reshape(self.block_group_info_len, self.max_depth, self.kernel_list_len), expand_ratio.reshape(self.block_group_info_len, self.max_depth, self.expand_ratio_list_len), depth.reshape(self.block_group_info_len, self.depth_list_len), resolution
    
class MetaHyperNetwork(nn.Module):
    def __init__(self, block_group_info_len, max_depth, kernel_list_len, expand_ratio_list_len, depth_list_len, resolution_len, num_random_devices=50, hw_embed_dim=10, hpn_type="base", use_zero_init=False, use_softmax=True):
        super(MetaHyperNetwork, self).__init__()
        self.hardware_embedding = nn.Embedding(num_random_devices, hw_embed_dim)
        self.hpn_factory = [hpn_type(block_group_info_len, max_depth, kernel_list_len, expand_ratio_list_len, depth_list_len, resolution_len, use_zero_init=use_zero_init, use_softmax=use_softmax) for i in range(num_random_devices)]
        self.hpn_factory = nn.ModuleList(self.hpn_factory)

        self.hw_embed_dim = hw_embed_dim
        self.block_group_info_len = block_group_info_len
        self.max_depth = max_depth
        self.kernel_list_len = kernel_list_len
        self.expand_ratio_list_len = expand_ratio_list_len
        self.depth_list_len = depth_list_len
        self.resolution_len = resolution_len

    def forward(self, x, hw):
        similarity = torch.squeeze(torch.matmul(hw, self.hardware_embedding.weight.transpose(0,1)))/torch.sqrt(torch.tensor(self.hw_embed_dim*1.0))
        similarity = torch.softmax(similarity, dim=-1)
        out_kernel = torch.zeros(self.block_group_info_len, self.max_depth, self.kernel_list_len).to(x.device)
        out_expand_ratio = torch.zeros(self.block_group_info_len, self.max_depth, self.expand_ratio_list_len).to(x.device)
        out_depth = torch.zeros(self.block_group_info_len, self.depth_list_len).to(x.device)
        out_resolution = torch.zeros(self.resolution_len).to(x.device)
        for i in range(similarity.shape[0]):
            hpn_out_kernel, hpn_out_expand_ratio, hpn_out_depth, hpn_out_resolution = self.hpn_factory[i](x)
            out_kernel+= similarity[i]*hpn_out_kernel
            out_expand_ratio+= similarity[i]*hpn_out_expand_ratio
            out_depth+= similarity[i]*hpn_out_depth
            out_resolution+= similarity[i]*torch.squeeze(hpn_out_resolution)
        return [out_kernel, out_expand_ratio, out_depth, out_resolution]

# test ofa hypernetwork
'''input = torch.tensor([[0.2,0.8]])
hw = torch.randn(1,10)
mhn = MetaHyperNetwork(3, 4, 3, 3, 3, 50, 10, "attention")
out_kernel, out_expand_ratio, out_depth = mhn(input, hw)
print(out_kernel.shape)
print(out_expand_ratio.shape)
print(out_depth.shape)'''