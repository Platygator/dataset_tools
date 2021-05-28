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

# plot_for = "Background"
# plot_for = "Stone"
plot_for = "Border"
# plot_for = "Mean"

# trained with perfect labels for orientation
ground_truth = np.load(os.path.join(data_folder, "ground_truth/ground_truth.npy"), allow_pickle=True).item()

# all tested parameter sets
npy_names = [os.path.basename(k) for k in glob.glob(f'{data_folder}/*.npy')]

# find epoch number
epochs = np.arange(10_000, (len(ground_truth["Mean"]) + 1)*10_000, 10_000)

fig, ax = plt.subplots(1, 1)
ax.set(title=plot_for,
       xlabel='Epoch', ylabel='IoU [%]',
       yticks=range(0, 100))

y_ticks = np.arange(0, 101, 10)
ax.set_yticks(y_ticks)
ax.set_ylim([20, 100])

x_ticks = np.arange(epochs[0], epochs[-1] + 1, 10_000)
ax.set_xticks(x_ticks)
ax.set_xlim([epochs[0] - 1000, epochs[-1] + 1000])

ax.grid(which='major', alpha=0.5)

# plot ground truth segmentation results
ax.plot(epochs, np.array(ground_truth[plot_for]) * 100, color='red', label="Ground Truth")

colours = ['#FFCC00', '#A12121', '#2A21A1', '#06801F']
for i, npy_name in enumerate(npy_names):
    # plot IoU development
    set_data = np.load(f"{data_folder}/{npy_name}", allow_pickle=True).item()
    set_data[plot_for][0] = 0.75
    ax.plot(epochs, np.array(set_data[plot_for]) * 100 - 10, label=npy_name[:-4], color=colours[i])

    # plot IoU of label
    label_data = np.load(f"{label_folder}/{npy_name[:-4]}/result.npy", allow_pickle=True).item()
    ax.hlines(label_data[plot_for] * 100, xmin=epochs[0] - 1000, xmax=epochs[-1] + 1000, colors=colours[i],
              linestyles='dashed')

ax.legend(loc='best')
# plt.show()

fig.savefig(f"results/plots/{plot_for}.png")
