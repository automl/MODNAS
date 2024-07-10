import torch.nn as nn
import torch
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num, hw_embed_on, hw_embed_dim):
        super(Net, self).__init__()
        self.hw_embed_on = hw_embed_on

        self.first_layer = nn.Linear(feature_dim, hidden_dim).cuda()
        if hw_embed_on:
            self.fc_hw1 = nn.Linear(hw_embed_dim, hidden_dim).cuda()
            self.fc_hw2 = nn.Linear(hidden_dim, hidden_dim).cuda()

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim).cuda())
        self.third_last_layer = nn.Linear(2*hidden_dim, hidden_dim).cuda()
        self.second_last_layer = nn.Linear(hidden_dim, hidden_dim).cuda()

        self.predict = nn.Linear(hidden_dim, 1).cuda()
        #self.scale = nn.Parameter(torch.ones(1)*20)

    def forward(self, x, hw_embed=None):
        #hw_embed = hw_embed.half()
        #x = x.half()
        hw_embed = hw_embed.float()
        x = torch.squeeze(x)
        hw_embed = torch.squeeze(hw_embed)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(hw_embed.shape) == 1:
            hw_embed = hw_embed.unsqueeze(0)
        #print(x.shape)
        #print(hw_embed.shape)
        if self.hw_embed_on:
            hw_embed = hw_embed.repeat(x.shape[0], 1)
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
        
        if self.hw_embed_on:
            hw = F.relu(self.fc_hw1(hw_embed.to(x.device)))
            hw = F.relu(self.fc_hw2(hw))
            #print(x.shape)
            #print(hw.shape)
            x = torch.cat((x, hw), dim=-1)

        x = F.relu(self.third_last_layer(x))
        x = F.relu(self.second_last_layer(x))

        x = self.predict(x)#*self.scale

        return x