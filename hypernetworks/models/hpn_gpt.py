import torch.nn as nn
import torch
class HyperNetwork(nn.Module):
    def __init__(self, choices):
        super(HyperNetwork, self).__init__()
        num_layers = max(choices['n_layer_choices'])
        num_head_choices = len(choices['n_head_choices'])
        num_mlp_ratio_choices = len(choices['mlp_ratio_choices'])
        num_embedding_dim_choices = len(choices['embed_dim_choices'])
        num_layers_choices = len(choices['n_layer_choices'])
        self.num_layers = num_layers
        self.num_head_choices = num_head_choices
        self.num_mlp_ratio_choices = num_mlp_ratio_choices
        self.num_embedding_dim_choices = num_embedding_dim_choices
        self.num_layers_choices = num_layers_choices
        self.embedding_layer = nn.Embedding(101, num_layers_choices)
        self.embedding_head = nn.Embedding(101, num_head_choices*num_layers)
        self.embedding_mlp_ratio = nn.Embedding(101, num_mlp_ratio_choices*num_layers)
        self.embedding_embedding_dim = nn.Embedding(101, num_embedding_dim_choices)
        self.embedding_bias_choice = nn.Embedding(101, 2)

    def forward(self, x):
        id = int(x[0][0]*101)
        layer_arch = self.embedding_layer(torch.tensor([id]).to(x.device))
        head_arch = self.embedding_head(torch.tensor([id]).to(x.device))
        mlp_ratio_arch = self.embedding_mlp_ratio(torch.tensor([id]).to(x.device))
        embedding_dim_arch = self.embedding_embedding_dim(torch.tensor([id]).to(x.device))
        bias_choice = self.embedding_bias_choice(torch.tensor([id]).to(x.device))
        layer_arch = layer_arch.reshape(self.num_layers_choices)
        head_arch = head_arch.reshape(self.num_layers, self.num_head_choices)
        mlp_ratio_arch = mlp_ratio_arch.reshape(self.num_layers, self.num_mlp_ratio_choices)
        embedding_dim_arch = embedding_dim_arch.reshape(self.num_embedding_dim_choices)
        bias_choice = bias_choice.reshape(2)
        return layer_arch, head_arch, mlp_ratio_arch, embedding_dim_arch, bias_choice
    

class MetaHyperNetwork(nn.Module):
    def __init__(self, choices, num_random_devices=50, hw_embed_dim=10, hpn_type=HyperNetwork):
        super(MetaHyperNetwork, self).__init__()
        self.hardware_embedding = nn.Embedding(num_random_devices, hw_embed_dim)#.half()
        #if hpn_type == "base":
        self.hpn_factory = [hpn_type(choices) for i in range(num_random_devices)]
        #self.hpn_factory = [hpn_type(block_group_info_len, max_depth, kernel_list_len, expand_ratio_list_len, depth_list_len, resolution_len, use_zero_init=use_zero_init, use_softmax=use_softmax) for i in range(num_random_devices)]
        self.hpn_factory = nn.ModuleList(self.hpn_factory)

        self.hw_embed_dim = hw_embed_dim

    def forward(self, x, hw):
        # compute similarity between hw and all random devices
        hw = hw.float()
        hw = torch.unsqueeze(hw, 0)
        #print(hw.device)
        #print(self.hardware_embedding.weight.device)
        #print(x.device)
        similarity = torch.squeeze(torch.matmul(hw, self.hardware_embedding.weight.to(x.device).transpose(0,1)))/torch.sqrt(torch.tensor(self.hw_embed_dim*1.0).to(x.device))
        similarity = torch.softmax(similarity, dim=-1)
        out_embed = torch.zeros([self.hpn_factory[0].num_embedding_dim_choices]).to(x.device)
        out_layer = torch.zeros([self.hpn_factory[0].num_layers_choices]).to(x.device)
        out_head = torch.zeros([self.hpn_factory[0].num_layers, self.hpn_factory[0].num_head_choices]).to(x.device)
        out_mlp_ratio = torch.zeros([self.hpn_factory[0].num_layers, self.hpn_factory[0].num_mlp_ratio_choices]).to(x.device)
        out_bias = torch.zeros([2]).to(x.device)
        
        for i in range(similarity.shape[0]):
            layer_arch, head_arch, mlp_ratio_arch, embedding_dim_arch, bias_choice = self.hpn_factory[i](x)
            #print(hpn_out.shape)
            out_embed+= similarity[i]*torch.squeeze(embedding_dim_arch)
            out_layer+= similarity[i]*torch.squeeze(layer_arch)
            out_head+= similarity[i]*torch.squeeze(head_arch)
            out_mlp_ratio+= similarity[i]*torch.squeeze(mlp_ratio_arch)
            out_bias+= similarity[i]*torch.squeeze(bias_choice)
        return out_layer, out_head, out_mlp_ratio, out_embed, out_bias

if __name__ == "__main__":
    from predictors.gpt.hw_loader import search_spaces, HWDataset
    choices = search_spaces["s"]
    mhn = MetaHyperNetwork(choices).cuda()
    dataset = HWDataset()
    batch = dataset.sample_batch("a100", 64, "train")
    out = mhn(torch.tensor([[0.2,0.8]]).cuda(), batch[2].float().cuda())
    for i in out:
        print(i.shape)
    # pretrain
    '''dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0, 1.0]))
    optimizer = torch.optim.Adam(mhn.parameters(), lr=1e-3)
    for i in range(100):
        for j in range(1000):
            scalarization = dirichlet.sample().cuda().unsqueeze(0)
            hw_emb = dataset.sample_batch("a100", 1, "train")[2].float().cuda()
            out = mhn(scalarization, hw_emb)
            rand_out = [torch.randn_like(out[0])*1e-3, torch.randn_like(out[1])*1e-3, torch.randn_like(out[2])*1e-3, torch.randn_like(out[3])*1e-3, torch.randn_like(out[4])*1e-3]
            loss = sum([torch.nn.MSELoss()(out[i], rand_out[i]) for i in range(5)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j%100==0:
                print("Loss:", loss.item())
                print(torch.nn.functional.softmax(rand_out[0],dim=-1))
                #print(out)
                save_path = "pretrained_hpns/mhn_gpt.pth"
                torch.save(mhn.state_dict(), save_path)'''