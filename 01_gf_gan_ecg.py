import argparse
import datetime
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import winsound
import json
import torch.nn as nn
import torch
from ekg_class import dicts
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path
from scipy.signal import resample
import torch.nn.functional as F


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("\n*****start time:      {0:02d}:{1:02d}:{2:02.0f}".format(
        start_time.hour, start_time.minute, start_time.second))
sim = "01"
# path_base = "/home/jprevost/pycharm_proj_edmond/01/"
path_base = "F:\\UTSA\\ECG_Synthesis\\dell_g7"
path_read = "\\Datasets\\mitbih_datasets_Dictionaries\\"
path_write = "\\PycharmProjects\\Github_paper1\\run_7\\"
Path(path_base + path_write).mkdir(parents=False, exist_ok=True)
Path(path_base + path_write + sim + "_gen_beats_run_7/").mkdir(parents=False, exist_ok=True)
Path(path_base + path_write + sim + "_errors").mkdir(parents=True, exist_ok=True)

resampled_to = 256

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",   type=int,   default=30,     help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=9,      help="size of the batches")
parser.add_argument("--lr",         type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1",         type=float, default=0.5,    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",         type=float, default=0.999,  help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu",      type=int,   default=8,      help="# of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=100,    help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=400, help="interval")
opt = parser.parse_args()

batch_size = opt.batch_size
print("opt:\n\t{}".format(opt))
cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        """
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

            # *block(128, 256),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # *block(256, 512),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # *block(512, 1024),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, resampled_to),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(resampled_to, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, beat):
        out = self.model(beat)

        return out


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Dataset and Configure data loader
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
    if item[2] in classes_to_keep_letter:
        X.append(item[1])
        y.append(letter2num[item[2]])

# *********************  End of ECG Dataset **************************
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------------------------------------------------------------------------------
#                                     Training
# ----------------------------------------------------------------------------------


d_loss_list = []
g_loss_list = []

# Adversarial ground truths
valid_label = Tensor(opt.batch_size, 1).fill_(1.0)
fake_label = Tensor(opt.batch_size, 1).fill_(0.0)
with open(path_base + path_write + sim + "_errors\\" + sim + '_loss_D.json', 'w', encoding='utf-8') as f:
    json.dump("[", f, ensure_ascii=False, indent=4)
with open(path_base + path_write + sim + "_errors/" + sim + '_loss_G.json', 'w', encoding='utf-8') as f:
    json.dump("[", f, ensure_ascii=False, indent=4)

#  opt.n_epochs = 1
for epoch in range(opt.n_epochs):
    with open('ecg_y.json') as f:
        y = json.load(f)
    gen_beats_epoch = []
    d_loss_epoch = []
    g_loss_epoch = []
    with open(path_base + path_write + sim + "_gen_beats_run_7/" + sim + '_gen_beats_ep{}.json'.
            format(epoch), 'w', encoding='utf-8') as f:
        json.dump("[", f, ensure_ascii=False, indent=4)
    for i, (beats, _) in enumerate(dataloader):
        a = i
        # Configure input
        real_beats = beats.float().cuda()

        # ----------------------------------------------------------------------------------
        #                                   Train Generator
        # ----------------------------------------------------------------------------------

        optimizer_G.zero_grad()

        # Sample noise as generator input, latent space
        z = Tensor(np.random.normal(0, 1, (beats.shape[0], opt.latent_dim)))

        # Generate a batch of beats
        fake = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(fake), valid_label)

        g_loss.backward()
        optimizer_G.step()

        # ----------------------------------------------------------------------------------
        #                                   Train Discriminator
        # ----------------------------------------------------------------------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_beats), valid_label)
        fake_loss = adversarial_loss(discriminator(fake.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ******************  distances (beginning) ******************* ea
        d_loss_epoch.extend([d_loss.item()])
        g_loss_epoch.extend([g_loss.item()])
        gen_beats_epoch.extend([fake.tolist()])
        # ******************     distances (end)     ******************* ea

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch {:4d}/{:3d}] [Batch {:5d}/{:5d}] [D loss: {:10.7f}] [G loss: {:10.7f}]".format(
                    epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        b = 0

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
# winsound.Beep(frequency, duration)

frequency = 262  # Set Frequency To 2500 Hz
duration = 1200  # Set Duration To 1000 ms == 1 second
# winsound.Beep(frequency, duration)
