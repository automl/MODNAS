# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import random
import configargparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from search_spaces.hat.fairseq import utils

def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency

def sample_one_choice_from_config(config):
    sampled_config = {}
    for key in config:
        if isinstance(config[key], list):
            # sample list index
            idx = np.random.randint(0, len(config[key]))
            sampled_config[key] = config[key][idx]
        else:
            sampled_config[key] = config[key]
    return sampled_config

# add early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
      self.patience = patience
      self.verbose = verbose
      self.counter = 0
      self.best_score = None
      self.early_stop = False
      self.val_loss_min = np.Inf
      self.delta = delta
      self.path = path
      self.trace_func = trace_func

    def __call__(self, val_loss, model):

      score = val_loss

      if self.best_score is None:
          self.best_score = score
          self.save_checkpoint(val_loss, model)
      elif score < self.best_score + self.delta:
          self.counter += 1
          self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
          if self.counter >= self.patience:
              self.early_stop = True
      else:
          self.best_score = score
          self.save_checkpoint(val_loss, model)
          self.counter = 0

    def save_checkpoint(self, val_loss, model):
      '''Saves model when validation loss decrease.'''
      if self.verbose:
          self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
      torch.save(model.state_dict(), self.path)
      self.val_loss_min = val_loss

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
        self.scale = nn.Parameter(torch.ones(1)*20)

    def forward(self, x, hw_embed=None):
        #hw_embed = hw_embed.half()
        #x = x.half()
        x = x.float()
        hw_embed = hw_embed.float().cuda()
        x = torch.squeeze(x).cuda()
        hw_embed = torch.squeeze(hw_embed)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(hw_embed.shape) == 1:
            hw_embed = hw_embed.unsqueeze(0)
        #print(x.shape)
        #print(hw_embed.shape)
        if self.hw_embed_on:
            hw_embed = hw_embed.repeat(x.shape[0], 1)
        #print(x)
        #print(self.first_layer.weight)
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


class LatencyPredictor(object):
    def __init__(self, feature_norm, lat_norm, ckpt_path, lat_dataset_dicts={}, feature_dim=10, hidden_dim=400, hidden_layer_num=3, train_steps=5000, bsz=128, lr=1e-5, search_space="space1", task="wmt14.en-de"):
        self.lat_datset_dicts = lat_dataset_dicts
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.task = task
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.search_space = search_space
        self.ckpt_path = ckpt_path
        self.keys_train , self.keys_test = self.get_train_test_devices()

        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num, hw_embed_on=True, hw_embed_dim=10).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=50, verbose=True, path=self.ckpt_path,delta=0.0001)

        self.train_steps = train_steps
        self.bsz = bsz
        self.compute_lats_vector_dict()

    def sample_batch(self, dataset):
        # randomly sample batch
        sample_x = []
        sample_y = []
        for i in range(self.bsz):
            # sample random index from the list of architectures
            index = np.random.randint(0, len(dataset))
            sample_dict = sample_one_choice_from_config(dataset[index])
            sample_x.append(utils.get_config_features_one_hot(sample_dict, search_space=self.search_space))
            sample_y.append((dataset[index]["latency_mean_encoder"]+dataset[index]["latency_mean_decoder"])/self.lat_norm)
        return torch.tensor(sample_x), sample_y
    
    def get_train_test_devices(self):
        task_name = self.task
        if task_name == "wmt14.en-de" or task_name == "wmt14.en-fr":
            train_devices = ["cpu_xeon", "gpu_titanxp"]
            test_devices = ["cpu_raspberrypi"]
        else:
            train_devices = ["gpu_titanxp"]
            test_devices = ["gpu_titanxp"]
        return train_devices, test_devices

    def train(self):
        self.model = self.model.cuda()
        for i in range(self.train_steps):
            # randomly sample device
          for j in range(100):
            sample_device = random.choice(self.keys_train)
            if self.task == "iwslt14.de-en" or self.task == "wmt19.en-de":
                train_dataset = self.lat_datset_dicts[sample_device][:1900]
            else:
                train_dataset = self.lat_datset_dicts[sample_device]
            train_hw_embed = self.lats_vector_dict[sample_device]
            # randomly sample batch
            sample_x, sample_y = self.sample_batch(train_dataset)
            sample_hw_embed = train_hw_embed

            sample_x_tensor = sample_x
            sample_y_tensor = torch.Tensor(sample_y).cuda()
            sample_hw_embed_tensor = torch.Tensor(sample_hw_embed).cuda()

            prediction = self.model(sample_x_tensor.cuda(),sample_hw_embed_tensor).squeeze()
            from scipy.stats import spearmanr
            if j == 99:
             corr_sr = spearmanr(self.lat_norm*prediction.cpu().detach().numpy(), self.lat_norm*sample_y_tensor.cpu().numpy())
             print(f"Spearman correlation: {corr_sr.correlation}")
            loss = self.criterion(prediction, sample_y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            # grad clipping
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # validation
          if i % 5 == 0:
                with torch.no_grad():
                    sample_device = random.choice(self.keys_test)
                    if self.task == "iwslt14.de-en" or self.task == "wmt19.en-de":
                        test_dataset = self.lat_datset_dicts[sample_device][1900:]
                    else:
                        test_dataset = self.lat_datset_dicts[sample_device]
                    test_hw_embed = self.lats_vector_dict[sample_device]
                    # randomly sample batch
                    sample_x, sample_y = self.sample_batch(test_dataset)
                    sample_y = torch.Tensor(sample_y)
                    sample_hw_embed = test_hw_embed
                    prediction = self.model(sample_x.cuda(),torch.Tensor(sample_hw_embed).cuda()).squeeze().cpu()
                    loss = self.criterion(prediction.cuda(), sample_y.cuda())
                    #print(f"Predicted latency: {prediction}")
                    #print(f"Real latency: {sample_y}")
                    print(f"Loss: {loss}")
                    print(f"RMSE: {np.sqrt(self.criterion(self.lat_norm*sample_y.cpu(), self.lat_norm*prediction.cpu()))}")
                    print(f"MAPD: {torch.mean(torch.abs((sample_y.cpu() - prediction.cpu()) / sample_y.cpu()))}")
                    # compute spearman correlation
                    from scipy.stats import spearmanr
                    corr_sr = spearmanr(self.lat_norm*prediction.cpu().numpy(), self.lat_norm*sample_y.cpu().numpy())
                    print(f"Spearman correlation: {corr_sr.correlation}")
                    from scipy.stats import kendalltau
                    corr = kendalltau(self.lat_norm*prediction.cpu().numpy(), self.lat_norm*sample_y.cpu().numpy())
                    print(f"Kendalltau correlation: {corr.correlation}")

                self.early_stopping(corr_sr.correlation, self.model)
          if self.early_stopping.early_stop:
                print("Early stopping")
                break


    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config, space="space0"):
        with torch.no_grad():
            features = utils.get_config_features_one_hot(config,"space1")
            if self.task == "iwslt14.de-en" or self.task == "wmt19.en-de":
                test_dataset = self.lats_vector_dict["gpu_titanxp"]
            else:
                test_dataset = self.lats_vector_dict["cpu_raspberrypi"]
            test_hw_embed = torch.tensor(test_dataset).squeeze().unsqueeze(0)
            hw_embed = torch.tensor(test_hw_embed).cuda().float().squeeze().unsqueeze(0)
            prediction = self.model(torch.Tensor(features).float().cuda(),hw_embed.cuda()).squeeze().cpu()*self.lat_norm

        return prediction

    def split(self):
        # select held out test device randomly
        test_device = "cpu_raspberrypi"
        print(f"Test device: {test_device}")

        self.test_x = self.dataset[test_device]['x']
        self.test_y = self.dataset[test_device]['y']
        self.test_hw_embed = self.dataset[test_device]['lats_vector']
        # delete test device from dataset
        del self.dataset[test_device]
        # checks 
        for k in self.dataset.keys():
          print(f"Device: {k}, # samples: {len(self.dataset[k]['x'])}")

    def compute_lats_vector_dict(self):
        lats_vector_dict = {}
        for k in self.lat_datset_dicts.keys():
            list_archs = self.lat_datset_dicts[k]
            archs_lats = []
            for arch in list_archs:
                archs_lats.append((arch["latency_mean_encoder"]+arch["latency_mean_decoder"])/self.lat_norm)
            arch_ids_for_hw = list(range(len(archs_lats)))
            arch_inds_emb = arch_ids_for_hw[::len(arch_ids_for_hw)//10]
            lats_vector = [archs_lats[i] for i in arch_inds_emb]

            lats_vector_dict[k] = normalization(np.array(lats_vector),portion=1.0)
        self.lats_vector_dict = lats_vector_dict
    
    def read_all_datasets(self, search_space="space0"):
        self.dataset = {}
        for path in self.dataset_path:
          device = path.split('/')[-1][:-4]
          device = device.split('_')[-2]+'_'+path.split('/')[-1].split('_')[-1]
          features_norm_all = []
          lats_all = []
          with open(path,"rb") as f:
               configs_list = pickle.load(f)
          with open(path, 'r') as fid:
            next(fid) # skip first line of CSV
            for line in fid:
                features = line.split(',')[:self.feature_dim]
                features_eval = list(map(eval, features))
                #features_norm = np.array(features_eval) / self.feature_norm
                features_norm = utils.get_config_features_one_hot_from_vec(features_eval, search_space)
                features_norm = np.array(features_norm)
                print(features_norm)
                features_norm_all.append(features_norm)

                lats = line.split(',')[self.feature_dim:]
                total_lat = eval(lats[0]) + eval(lats[1])
                lats_all.append(total_lat / self.lat_norm)
          tmp = list(zip(features_norm_all, lats_all))
          # uniformly sample 10 sample ids from each device
          arch_ids_for_hw = list(range(len(tmp)))
          # sample 10 archs at uniform intervals
          arch_ids_for_hw = arch_ids_for_hw[::len(arch_ids_for_hw)//10]
          lats_vector = [lats_all[i] for i in arch_ids_for_hw]
          lats_vector_norm = normalization(np.array(lats_vector),portion=1.0)
          random.shuffle(tmp)
          features_norm_all, lats_all = zip(*tmp)
          self.dataset[device] = {'x': features_norm_all, 'y': lats_all, 'lats_vector': lats_vector_norm}
        # checks 
        for k in self.dataset.keys():
          print(f"Device: {k}, # samples: {len(self.dataset[k]['x'])}")



def get_lat_paths_for_task(task):
    
    if task == "wmt14.en-de":
        device_paths_dict = {}
        device_paths_dict["cpu_raspberrypi"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14ende_cpu_raspberrypi.pkl"
        device_paths_dict["cpu_xeon"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14ende_cpu_xeon.pkl"
        device_paths_dict["gpu_titanxp"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14ende_gpu_titanxp.pkl"
        # load pickle files
        for device in device_paths_dict:
            with open(device_paths_dict[device], "rb") as f:
                device_paths_dict[device] = pickle.load(f)
        search_space = "space0"
    
    elif task == "wmt14.en-fr":
        device_paths_dict = {}
        device_paths_dict["cpu_raspberrypi"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14enfr_cpu_raspberrypi.pkl"
        device_paths_dict["cpu_xeon"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14enfr_cpu_xeon.pkl"
        device_paths_dict["gpu_titanxp"] = "hypernetwork_data_utils/hat//config_to_lat_list_wmt14enfr_gpu_titanxp.pkl"
        for device in device_paths_dict:
            with open(device_paths_dict[device], "rb") as f:
                # read silently
                device_paths_dict[device] = pickle.load(f)
        search_space = "space0"



    return device_paths_dict, search_space

if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str, help='task name')

    parser.add_argument('--lat-dataset-paths', type=str, default='./latency_dataset/lat.tmp', help='the paths to read latency dataset')
    parser.add_argument('--feature-norm', type=float, nargs='+', default=[640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2], help='normalizing factor for each feature')
    parser.add_argument('--lat-norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--feature-dim', type=int, default=101, help='dimension of feature vector')
    parser.add_argument('--hidden-dim', type=int, default=400, help='hidden dimension of FC layers in latency predictor')
    parser.add_argument('--hidden-layer-num', type=int, default=6, help='number of FC layers')
    parser.add_argument('--ckpt-path', type=str, default='latency_dataset/ckpts/tmp.pt', help='path to save latency predictor weights')
    parser.add_argument('--train-steps', type=int, default=5000, help='latency predictor training steps')
    parser.add_argument('--bsz', type=int, default=128, help='latency predictor training batch size')
    parser.add_argument('--lr', type=float, default=1e-6, help='latency predictor training learning rate')

    args = parser.parse_args()
    args.lat_dataset_dicts, search_space = get_lat_paths_for_task(args.task)
    ckpt_path   = "predictor_data_utils/hat/"+args.task+"_one_hot_"+".pt"
    predictor = LatencyPredictor(lat_dataset_dicts=args.lat_dataset_dicts,
                           feature_norm=args.feature_norm,
                           lat_norm=args.lat_norm,
                           feature_dim=args.feature_dim,
                           hidden_dim=args.hidden_dim,
                           hidden_layer_num=args.hidden_layer_num,
                           ckpt_path=ckpt_path,
                           train_steps=args.train_steps,
                           bsz=args.bsz,
                           lr=args.lr,
                           search_space=search_space,
                           task = args.task)

    #predictor.read_all_datasets(search_space=search_space)
    #predictor.split()
    predictor.train()
    print('Latency predictor training finished')

    predictor.load_ckpt()
    config_example = {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 6,
            'encoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072, 3072],
            'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4],
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 3072, 3072, 3072, 1024],
            'decoder_self_attention_heads': [4, 8, 8, 4, 4],
            'decoder_ende_attention_heads': [4, 8, 8, 4, 4],
            'decoder_arbitrary_ende_attn':  [-1, 1, 1, 1, 1]
    }

    predict = predictor.predict_lat(config_example)
    print(f'Example config: {config_example}')
    print(f'Example latency: {predict}')