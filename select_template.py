import json
from matplotlib import pyplot as plt

mypath_r = "/home/jprevost/pycharm_proj_edmond/ClearLab_alien/"
fname = "ecg_X_normalized.json"
with open(mypath_r + fname, "r") as f:
    X = json.load(f)

with open(mypath_r + "ecg_y.json", "r") as f:
    y = json.load(f)
"""
for i in range(len(X)):
    plt.close()
    fig = plt.figure("Template, Normal Beat, i {}, Class {}".format(i, y[i]))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X[i])
    ax.grid()
    plt.savefig(mypath_r + "/dataset_plots/" + "beat_{}_{}".format(y[i], i))
"""

# with open(mypath_r + "template_index_132.json", "w") as f:
#    json.dump(X[132], f)

with open("/home/jprevost/pycharm_proj_edmond/run_6/01_genbeats/01_gen_beats_ep0.json", "r") as f:
    y = json.load(f)
a = 0