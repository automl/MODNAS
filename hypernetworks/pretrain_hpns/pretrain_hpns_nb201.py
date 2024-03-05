from hypernetworks.models.hpn_nb201 import MetaHyperNetwork, HyperNetwork
import torch
from predictors.help.loader import Data
help_loader = Data(mode='meta_test',
                     data_path='datasets/help/nasbench201',
                     search_space='nasbench201',
                     meta_train_devices=['1080ti_1',
                                         '1080ti_32',
                                         '1080ti_256',
                                         'silver_4114',
                                         'silver_4210r',
                                         'samsung_a50',
                                         'pixel3',
                                         'essential_ph_1',
                                         'samsung_s7'],
                     meta_valid_devices=['titanx_1',
                                         'titanx_32',
                                         'titanx_256',
                                         'gold_6240'],
                     meta_test_devices=['titan_rtx_256',
                                        'gold_6226',
                                        'fpga',
                                        'pixel2',
                                        'raspi4',
                                        'eyeriss'],
                     num_inner_tasks=8,
                     num_meta_train_sample=900,
                     num_sample=10,
                     num_query=1000,
                     sampled_arch_path=\
                     'datasets/help/nasbench201/arch_generated_by_metad2a.txt'
                    )
hpn_to_pretrain = []
hpn_names = []
devices_all = help_loader.meta_train_devices+help_loader.meta_valid_devices
for i in ['MetaHyperNetwork']:
    for j in ['HyperNetwork']:
        for k in [False]:
                if i == 'MetaHyperNetwork':
                    num_devices = 50
                    d = 10
                    
                else:
                    num_devices = 13
                    d = []
                    for device in devices_all:
                        d.append(help_loader.sample_specific_device_embedding( mode="meta-train", device = device)[0].unsqueeze(0))
                hpn = eval(i)(num_devices, d, eval(j), num_intervals=101)
                hpn_to_pretrain.append(hpn.cuda())
                hpn_names.append(i+j+str(k))

devices_all = help_loader.meta_train_devices+help_loader.meta_valid_devices
import os
os.makedirs("pretrained_hpns", exist_ok=True)
p = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*2))
mse_loss = torch.nn.MSELoss()
for k, hpn in enumerate(hpn_to_pretrain):
  optimizer = torch.optim.Adam(hpn.parameters(), lr=1e-3)
  for i in range(100):
     for j in range(1000):
          # sample scalarization 
          s = p.sample().reshape(1, -1).cuda()
          # sample device embedding 
          hw_emb,_ = help_loader.sample_device_embedding(mode="meta_train")
          hw_emb = hw_emb.reshape(1, -1).cuda()
          out_random = 1e-3*torch.randn(6,5).cuda()
          out = mse_loss(hpn(s, hw_emb), out_random)
          optimizer.zero_grad()
          out.backward()
          optimizer.step()
          optimizer.zero_grad()
          if j%100==0:
             print("Loss: ", out.item())
             print("Sample output softmax", torch.softmax(hpn(s, hw_emb), dim=-1))
             print("Out random softmax", torch.softmax(out_random, dim=-1))
             # save hpn
             torch.save(hpn.state_dict(), "pretrained_hpns/"+hpn_names[k]+".pth")



    
