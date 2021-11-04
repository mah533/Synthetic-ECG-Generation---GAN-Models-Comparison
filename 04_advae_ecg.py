"""
Model 04
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import itertools

# import winsound
# from scipy.signal import resample
from torch.optim import Adam

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch
# import gc

# from similaritymeasures import frechet_dist
# from scipy.spatial.distance import euclidean
# from dtaidistance.dtw import distance
# from sklearn.metrics import mutual_info_score as MI
from ekg_class import dicts

sim = "04"
path_base = "F:\\UTSA\\ECG_Synthesis\\dell_g7"
path_read = "\\Datasets\\mitbih_datasets_Dictionaries\\"
path_write = "\\PycharmProjects\\Github_paper1\\run_7\\"
Path(path_base + path_write).mkdir(parents=False, exist_ok=True)
Path(path_base + path_write + sim + "_gen_beats_run_7/").mkdir(parents=False, exist_ok=True)
Path(path_base + path_write + sim + "_errors").mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=9, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--beat_size", type=int, default=32, help="size of each beat length")
parser.add_argument("--channels", type=int, default=1, help="number of beat channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between beat sampling")
parser.add_argument('--dry_run', action='store_true', help='check a single training cycle works', default=True)
opt = parser.parse_args()
print(opt)

start_time = datetime.datetime.now()
print("\n*****start time:      {0:02d}:{1:02d}:{2:02.0f}".format(
    start_time.hour, start_time.minute, start_time.second))

resampled_to = 256

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim)))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(resampled_to, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, beat):
        x = self.model(beat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, resampled_to),
            nn.Tanh(),
        )

    def forward(self, z):
        beat = self.model(z)

        return beat


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.model(z)
        return out


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    l1_loss.cuda()

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

dataset     = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader  = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

# Optimizers
optimizer_G = Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

# euclid_list     = []
# dtw_list        = []
# frechet_list    = []

if opt.dry_run:
    opt.niter = 1

with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'w', encoding='utf-8') as f:
    json.dump("["[0], f, ensure_ascii=False, indent=4)
with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'w', encoding='utf-8') as f:
    json.dump("["[0], f, ensure_ascii=False, indent=4)

for epoch in range(opt.n_epochs):
    gen_beats_epoch = []
    d_loss_epoch = []
    g_loss_epoch = []
    for i, (beats, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid_label = Tensor(beats.shape[0], 1).fill_(1.0)
        fake_label  = Tensor(beats.shape[0], 1).fill_(0.0)

        # Configure input
        real_beats = beats.float().cuda()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_beats = encoder(real_beats)
        decoded_beats = decoder(encoded_beats)

        # Gen Loss
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_beats), valid_label) + 0.999 * l1_loss(
            decoded_beats, real_beats)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Tensor(np.random.normal(0, 1, (beats.shape[0], opt.latent_dim)))

        # Measure discriminator's ability to classify real from generated samples
        real_loss   = adversarial_loss(discriminator(z), valid_label)
        fake_loss   = adversarial_loss(discriminator(encoded_beats.detach()), fake_label)
        d_loss      = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        
        """
        ******************  distances (beginning) ******************* ea
        euclid_temp = []
        dtw_temp    = []
        frechet_temp = []

        batches_to_consider = 1
        fake = fake.squeeze(dim=1)
        y2 = np.array(templates)
        for batch in range(0, batches_to_consider):
             y1  = decoded_beats.cpu().detach().numpy()[batch, :]
             euclid_temp.extend([euclidean(y1, y2)])
             dtw_temp.extend([distance(y1, y2)])
             frechet_temp.extend([frechet_dist(y1, y2)])

        euclid_list.extend([euclid_temp])
        dtw_list.extend([dtw_temp])
        frechet_list.extend([frechet_temp])
        """
        d_loss_epoch.extend([d_loss.item()])
        g_loss_epoch.extend([g_loss.item()])

        gen_beats_epoch.append(decoded_beats.cpu().detach().tolist())
        # ******************     distances (end)     ******************* ea

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch {:3d}/{:3d}]    [Batch {:6d}/{:6d}]     [D loss: {:8.6f}]       [G loss: {:8.6f}]".
                    format(epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'a', encoding='utf-8') as f:
        json.dump(",", f, ensure_ascii=False, indent=4)
    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_D.json', 'a', encoding='utf-8') as f:
        json.dump(d_loss_epoch, f, ensure_ascii=False, indent=4)

    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'a', encoding='utf-8') as f:
        json.dump(",", f, ensure_ascii=False, indent=4)
    with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'a', encoding='utf-8') as f:
        json.dump(g_loss_epoch, f, ensure_ascii=False, indent=4)

    with open(path_base + path_write + sim + "_gen_beats_run_7/" + sim + '_gen_beats_ep{}.json'.format(epoch), 'a',
              encoding='utf-8') as f:
        json.dump(",", f, ensure_ascii=False, indent=4)

    with open(path_base + path_write + sim + "_gen_beats_run_7/" + sim + '_gen_beats_ep{}.json'.format(epoch), 'a',
              encoding='utf-8') as f:
        json.dump(gen_beats_epoch, f, ensure_ascii=False, indent=4)
    if opt.dry_run:
        break

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
print("\nstart time: {},\t finish: {}".format(start_time, datetime.datetime.now()))
print("total elapsed time: {}".format(elapsed_time))
print("elapsed time: \t  {0:02d}:{1:02.0f}:{2:02.0f}\n (hh:mm:ss)".format(hours, minutes, seconds))

# *******************************************************************************
#                               completion alarm
# *******************************************************************************

frequency = 440  # Set Frequency To 2500 Hz
duration = 1200  # Set Duration To 1000 ms == 1 second
# winsound.Beep(frequency, duration)

frequency = 262  # Set Frequency To 2500 Hz
duration = 1200  # Set Duration To 1000 ms == 1 second
# winsound.Beep(frequency, duration)
