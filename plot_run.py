"""
Created by Jan Schiffeler on 13.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

label_folder = "results"
data_folder = "mmseg_results"
run_name = "r50_umarked"

# all tested parameter sets
npy_name = f"{run_name}.npy"

fig, ax = plt.subplots(1, 1)
ax.set(title=run_name,
       xlabel='Epoch', ylabel='IoU [%]',
       yticks=range(0, 100))

y_ticks = np.arange(0, 101, 10)
ax.set_yticks(y_ticks)
ax.set_ylim([0, 100])


ax.grid(which='major', alpha=0.5)

epochs = np.arange(100, 601, 100)

# plot IoU development
set_data = np.load(f"{data_folder}/{npy_name}", allow_pickle=True).item()
ax.plot(epochs, np.array(set_data['Background']) * 100, label='Background', color='blue')
ax.plot(epochs, np.array(set_data['Stone']) * 100, label='Stone', color='orange')
ax.plot(epochs, np.array(set_data['Border']) * 100, label='Border', color='green')
ax.plot(epochs, np.array(set_data['Mean']) * 100, label='Mean', color='red')

# plot IoU of label
try:
    label_data = np.load(f"{label_folder}/{npy_name[:-4]}/result.npy", allow_pickle=True).item()
    # label_data = np.load("mmseg_results/real_2_sim_base.npy", allow_pickle=True).item()
    ax.hlines(np.array(label_data['Background']) * 100, linestyles='dashed', xmin=epochs[0], xmax=epochs[-1], color='blue')
    ax.hlines(np.array(label_data['Stone']) * 100, linestyles='dashed', xmin=epochs[0], xmax=epochs[-1], color='orange')
    ax.hlines(np.array(label_data['Border']) * 100, linestyles='dashed', xmin=epochs[0], xmax=epochs[-1], color='green')
    ax.hlines(np.array(label_data['Mean']) * 100, linestyles='dashed', xmin=epochs[0], xmax=epochs[-1], color='red')
except FileNotFoundError:
    pass

ax.legend(loc='best')
# plt.show()

fig.savefig(f"results/plots/{run_name}.png")
