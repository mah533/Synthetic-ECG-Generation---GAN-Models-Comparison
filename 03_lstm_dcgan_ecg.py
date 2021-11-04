"""
Model 03
"""


from __future__ import print_function
import argparse
import datetime
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import winsound
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset
from ekg_class import dicts
from matplotlib import pyplot as plt

sim = "03"
path_base = "F:\\UTSA\\ECG_Synthesis\\dell_g7"
path_read = "\\Datasets\\mitbih_datasets_Dictionaries\\"
path_write = "\\PycharmProjects\\Github_paper1\\run_7\\"
Path(path_base + path_write).mkdir(parents=False, exist_ok=True)
Path(path_base + path_write + sim + "_gen_beats_run_7/").mkdir(parents=False, exist_ok=True)
Path(path_base + path_write + sim + "_errors").mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, help='path to dataset', default='.')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=9, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works', default=False)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default=path_base, help='folder to output beats and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
# opt.dry_run = True
print(opt)

start_time = datetime.datetime.now()
print("\n*****start time:      {0:02d}:{1:02d}:{2:02.0f}".format(
        start_time.hour, start_time.minute, start_time.second))

resampled_to = 256

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"{}\"".format(opt.dataset))


num2descr, letter2num, letter2descr = dicts()
num2letter = dict(zip(letter2num.values(), letter2num.keys()))

with open(path_base + path_read + "record_X_y_adapt_win_bef075_aft075_Normalized.json", 'r') as f:
    data = json.load(f)

with open('training_template_index_132.json') as f:
    template = json.load(f)

# classes = [0,    4,   5,   8,   12]
# classes = ['N', 'V', 'F', 'S', 'Q']

classes_to_keep_letter = ['N']
classes_to_keep_num = [letter2num[i] for i in classes_to_keep_letter]

X = []
y = []
for item in data:
    if (item[2] in classes_to_keep_letter):
        X.append(item[1])
        y.append(letter2num[item[2]])
# *********************  End of ECG Dataset **************************
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, drop_last=True)
assert dataset

device  = torch.device("cuda:0"   if opt.cuda    else "cpu")
ngpu    = int(opt.ngpu)
nz      = int(opt.nz)
ngf     = int(opt.ngf)
ndf     = int(opt.ndf)
nc      = 1                         # for ecg data,  # TODO: double check
# num_sample_beat = 280


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers):
        super().__init__()
        self.n_layers   = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=inp_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, bidirectional=True,
                            batch_first=True).double()
        self.fc = nn.Linear(hidden_dim * 2, resampled_to).double()
        self.tanh = nn.Tanh()


    def forward(self, input):
        h_0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim, dtype=torch.double).to(device)
        c_0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim, dtype=torch.double).to(device)
        output, (h_o, c_o) = self.lstm(input, (h_0, c_0))
        out = self.tanh(self.fc(output[:, -1, :]))

        return out.reshape((out.size(0), 1, out.size(-1)))

# TODO try larger number of cells in LSTM
netG = Generator(inp_dim=100, hidden_dim=1000, num_layers=2).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Linear(13, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #    output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        # else:
        output = self.model(input)

        return output.squeeze(dim=1).squeeze(dim=1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

# fixed_noise = torch.randn(opt.batchSize, nz, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# euclid_list     = []
# dtw_list        = []
# frechet_list    = []
# MI_list         = []
d_loss_list     = []
g_loss_list     = []

if opt.dry_run:
    opt.niter = 1
with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'w', encoding='utf-8') as f:
    json.dump("["[0], f, ensure_ascii=False, indent=4)
with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'w', encoding='utf-8') as f:
    json.dump("["[0], f, ensure_ascii=False, indent=4)

# opt.niter = 2
for epoch in range(opt.niter):
    gen_beats_epoch = []
    errD_epoch      = []
    errG_epoch      = []
    for i, data in enumerate(dataloader, 0):
        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
        ###############################################################
        # train with real
        netD.zero_grad()
        real_cpu    = data[0].to(device)
        real_cpu    = real_cpu.reshape(real_cpu.size(0), 1, resampled_to)
        batch_size  = real_cpu.size(0)
        label_Tensor = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

        output      = netD(real_cpu)
        errD_real   = criterion(output, label_Tensor)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise       = torch.randn(batch_size, 1, nz, device=device, dtype=torch.double)
        fake        = netG(noise)
        label_Tensor.fill_(fake_label)
        output      = netD(fake.float().detach())
        errD_fake   = criterion(output, label_Tensor)
        errD_fake.backward()
        D_G_z1      = output.mean().item()
        errD        = errD_real + errD_fake
        optimizerD.step()

        ###############################################
        # (2) Update G network: maximize log(D(G(z))) #
        ###############################################
        netG.zero_grad()
        label_Tensor.fill_(real_label)
        output = netD(fake.float())
        errG = criterion(output, label_Tensor)
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()
        """
        # ******************  distances (beginning) ******************* ea
        # euclid_temp     = []
        # dtw_temp        = []
        # frechet_temp    = []

        # batches_to_consider = 1
        fake = fake.squeeze(dim=1)
        # y2 = np.array(templates)
        # for batch in range(0, batches_to_consider):
            # y1 = fake.cpu().detach().numpy()[batch, :]
            # euclid_temp.extend([euclidean(y1, y2)])
            # dtw_temp.extend([distance(y1, y2)])
            # frechet_temp.extend([frechet_dist(y1, y2)])

        # euclid_list.extend([sum(euclid_temp) / batches_to_consider])
        #  euclid_list.extend([euclid_temp])
        # dtw_list.extend([sum(dtw_temp) / batches_to_consider])
        # dtw_list.extend([dtw_temp])
        # frechet_list.extend([sum(frechet_temp) / batches_to_consider])
        # frechet_list.extend([frechet_temp])
        # MI_list.extend([sum(MI_temp) / batches_to_consider])
        """

        errD_epoch.extend([errD.item()])
        errG_epoch.extend([errG.item()])
        gen_beats_epoch.append([fake.tolist()])
        # ******************     distances (end)     ******************* ea

        batches_done = epoch * len(dataloader) + i
        if i % 100 == 0:
            print('[{:3d}/{:3d}][{:5d}/{:5d}]   Loss_D: {:6.4f}   Loss_G: {:6.4f} '
                  '  D(x): {:6.4f}   D(G(z)): {:6.4f} / {:6.4f}'.
                  format(epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            """
            plt.close()
            fig = plt.figure("epoch: {}, batches done: {}".format(epoch, i))

            for axs in range(1, opt.batchSize + 1):
                ax = fig.add_subplot(3, 3, axs)
                ax.plot(fake[axs - 1].cpu().detach())
            plt.savefig("{}/03_ep{}_b{}.png".format(opt.outf, epoch, batches_done))
            """
        if opt.dry_run:
            break
    # do checkpointing
    # torch.save(netG.state_dict(), '{}/03_netG.pth'.format(opt.outf, epoch))
    # torch.save(netD.state_dict(), '{}/03_netD.pth'.format(opt.outf, epoch))

    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'a', encoding='utf-8') as f:
        json.dump(",", f, ensure_ascii=False, indent=4)
    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'a', encoding='utf-8') as f:
        json.dump(errD_epoch, f, ensure_ascii=False, indent=4)

    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'a', encoding='utf-8') as f:
        json.dump(",", f, ensure_ascii=False, indent=4)
    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'a', encoding='utf-8') as f:
        json.dump(errG_epoch, f, ensure_ascii=False, indent=4)

    with open(path_base + path_write + sim + "_gen_beats_run_7/" + sim + '_gen_beats_ep{}.json'.format(epoch), 'a',
              encoding='utf-8') as f:
        json.dump(",", f, ensure_ascii=False, indent=4)

    gbe = np.array(gen_beats_epoch).squeeze(axis=1).squeeze(axis=2).tolist()

    with open(path_base + path_write + sim + "_gen_beats_run_7/" + sim + '_gen_beats_ep{}.json'.format(epoch), 'a',
              encoding='utf-8') as f:
        json.dump(gbe, f, ensure_ascii=False, indent=4)

with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'a', encoding='utf-8') as f:
    json.dump("]", f, ensure_ascii=False, indent=4)

with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'a', encoding='utf-8') as f:
    json.dump("]", f, ensure_ascii=False, indent=4)

with open(path_base + path_write + sim + "_gen_beats_run_7/" + sim + '_gen_beats_ep{}.json'.format(epoch), 'a',
          encoding='utf-8') as f:
    json.dump("]", f, ensure_ascii=False, indent=4)

# calculation of "elapsed time"
elapsed_time = datetime.datetime.now() - start_time
hours, remainder = divmod(elapsed_time.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print("start time: {},\t finish: {}".format(start_time, datetime.datetime.now()))
print("total elapsed time: {}".format(elapsed_time))
print("elapsed time: \t  {0:02d}:{1:02.0f}:{2:02.0f}\n (hh:mm:ss)".format(hours, minutes, seconds))

# *******************************************************************************
#                               completion alarm
# *******************************************************************************

frequency = 440  # Set Frequency To 2500 Hz
duration = 1200  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

frequency = 262  # Set Frequency To 2500 Hz
duration = 1200  # Set Duration To 1000 ms == 1 second
# winsound.Beep(frequency, duration)
