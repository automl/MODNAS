from predictors.gpt.net import Net
from predictors.gpt.hw_loader import HWDataset, search_spaces
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import scipy
def train(model, gpus, train_loader, optimizer, epoch, log_interval=1000):
    model.train()
    for i in range(10000):
        random_gpu = np.random.choice(gpus)
        batch = train_loader.sample_batch(random_gpu, 1024, "train")
        arch = batch[0]
        latency = batch[1]
        hw_embed = batch[2]
        arch = arch.float().cuda()
        latency = latency.float().cuda()
        hw_embed = hw_embed.float().cuda()
        optimizer.zero_grad()
        output = model(arch, hw_embed)
        target = torch.squeeze(latency)
        output = torch.squeeze(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Loss: {:.6f}'.format(loss.item()))
            
def test(model, gpus, test_loader):
    model.eval()
    test_loss = 0
    out_test = []
    test_target = []
    with torch.no_grad():
        for i in range(1000):
            random_gpu = np.random.choice(gpus)
            batch = test_loader.sample_batch(random_gpu, 1024, "test")
            arch = batch[0]
            latency = batch[1]
            hw_embed = batch[2]
            arch = arch.float().cuda()
            latency = latency.float().cuda()
            target = torch.squeeze(latency).cuda()
            test_target.append(torch.squeeze(target))
            output = model(arch, hw_embed)
            output = torch.squeeze(output)
            out_test.append(torch.squeeze(output))
            test_loss += nn.MSELoss()(output, target).item()
    final_target = torch.cat(test_target,dim=-1)
    final_out = torch.cat(out_test,dim=-1)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    print("Corr", scipy.stats.kendalltau(final_target.cpu().numpy(), final_out.cpu().numpy())[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch HW Metric Predictor')
    parser.add_argument('--device', type=str, default='a100',
                        help='device name')
    parser.add_argument('--metric', type=str, default='energy_gpu',)
    parser.add_argument('--search_space', type=str, default='',
                        help='search space')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed ((default: 1)')
    parser.add_argument('--hw_embed_on', action='store_true', default=False,)
    parser.add_argument('--save_path', type=str, default='predictors/gpt/',
                        help='path to save the model checkpoints')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    #devices_all = ["P100","a6000", "rtx2080", "rtx3080", "v100", "a100", "a40", "h100", "cpu_mlgpu", "cpu_alldlc", "cpu_p100", "cpu_p100", "cpu_a6000", "cpu_meta", "helix_cpu"]
    models = ["", "m", "l"]
    gpus = ["a100","a6000", "rtx2080", "rtx3080", "v100", "P100", "a40", "h100"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataset = HWDataset(transform=None)
    # get the search space

    if args.search_space == "":
        ss = "s"
    else:
        ss = args.search_space
    choices_dict = search_spaces[ss]
    print(choices_dict)
    num_layers = max(choices_dict['n_layer_choices'])
    hw_embed_on = True
    hidden_dim = 256
    hiddent_layer_num = 4
    hw_embed_dim = 10
    hw_embed_on = True
    model = Net(80, hidden_dim, hiddent_layer_num, hw_embed_on, hw_embed_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_path = "predictors/gpt/" #+ "_hw_embed_on_" + str(hw_embed_on) + "_hw_embed_dim_" + str(hw_embed_dim) + "_layer_size_" + str(layer_size) + "_epochs_" + str(args.epochs) + "_lr_" + str(args.lr) + "_seed_" + str(args.seed) + ".pt"
    for epoch in range(1, args.epochs + 1):
        train(model, gpus, dataset, optimizer, epoch)
        test(model, gpus, dataset)
        torch.save(model.state_dict(), save_path+"metapredictor.pt")