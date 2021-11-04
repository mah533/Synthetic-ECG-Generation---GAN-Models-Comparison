"""
reads dtw, frechet, euclid dist. functions as a list for each model
"""


import json
import re
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

epoch_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
              "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
              "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]

myPath_base = "/home/jprevost/pycharm_proj_edmond/01/run_6/"

frac = 0.05
sim = "01"
dtw_list = []


myPath_errors = myPath_base + sim + "_errors/" + sim + "_distances"
with open(myPath_errors + "/" + sim + "_dtw_list_132" + ".json") as f:
    dtw_list = json.load(f)

with open(myPath_errors + "/" + sim + "_frechet_list_132" + ".json") as f:
    frechet_list = json.load(f)

with open(myPath_errors + "/" + sim + "_euclid_list_132" + ".json") as f:
    euclid_list = json.load(f)


t_vec = range(len(dtw_list))

dtw_fltd        = lowess(dtw_list, t_vec, frac=frac)[:, 1].tolist()
frechet_fltd    = lowess(frechet_list, t_vec, frac=frac)[:, 1].tolist()
euclid_fltd     = lowess(euclid_list, t_vec, frac=frac)[:, 1].tolist()

with open(myPath_errors + "/" + sim + "_dtw_fltd.json", "w") as f:
    json.dump(dtw_fltd, f)

with open(myPath_errors + "/" + sim + "_frechet_fltd.json", "w") as f:
    json.dump(frechet_fltd, f)

with open(myPath_errors + "/" + sim + "_euclid_fltd.json", "w") as f:
    json.dump(euclid_fltd, f)

"""
# myPath_root = "D:\\afghah_bkps\\gen_beats_run5\\"

xtick_labels = np.arange(1, 32, 2)
xticks = np.arange(0, 301650, 2 * 10055)

font = {'family': 'candara',
        'color': 'darkred',
        'size': 32,
        'fontstyle': 'italic',
        'fontweight': 'bold'
        }

gs = gridspec.GridSpec(1, 3)
fig = plt.figure("Generator and Discriminator Loss Functions and DTW Similarity Measures")
# fig.text(.5, .93, "Generator and Discriminator Loss Functions and DTW Similarity Measures", fontdict=font,
#         horizontalalignment="center", bbox=dict(facecolor='lightgray', alpha=0.8))
sigBig = 0
sigEnd = 301650
t_vec = range(sigBig, sigEnd)

ax_dtw = fig.add_subplot(gs[0, 0])
ax_dtw.grid()
ax_dtw.set_title('DTW Dist. Function')
ax_dtw.set_xticks(xticks)
ax_dtw.set_xticklabels(xtick_labels)

ax_lossD = fig.add_subplot(gs[0, 1])
ax_lossD.grid()
ax_lossD.set_title('Discriminator Loss')
ax_lossD.set_xticks(xticks)
ax_lossD.set_xticklabels(xtick_labels)

ax_lossG = fig.add_subplot(gs[0, 2])
ax_lossG.grid()
ax_lossG.set_title('Generator Loss')
ax_lossG.set_xticks(xticks)
ax_lossG.set_xticklabels(xtick_labels)


for i in range(1, 6, 1):
    sim = "0{}".format(i)
    myPath_errors = myPath_root + sim + "_errors\\"

    with open(myPath_errors + sim + "_dtw_fltd.json") as f:
        exec("dtw_fltd_0{} = json.load(f)".format(i))
    with open(myPath_errors + sim + "_lossD_fltd.json") as f:
        exec("lossD_fltd_0{} = json.load(f)".format(i))
    with open(myPath_errors + sim + "_lossG_fltd.json") as f:
        exec("lossG_fltd_0{} = json.load(f)".format(i))

    exec("ax_dtw.plot(t_vec, dtw_fltd_0{}[sigBig:sigEnd])".format(i))

    exec("ax_lossD.plot(t_vec, lossD_fltd_0{}[sigBig:sigEnd])".format(i))

    exec("ax_lossG.plot(t_vec, lossG_fltd_0{}[sigBig:sigEnd], label='0{}')".format(i, i))

ax_lossG.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
"""
