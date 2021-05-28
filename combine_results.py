"""
Created by Jan Schiffeler on 28.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import shutil
import os
import argparse


def create_argparser():
    parser = argparse.ArgumentParser(description='Chose label name')
    parser.add_argument('-l', '--label_set_name', required=True, help='label set name')
    args = vars(parser.parse_args())
    return args


arg = create_argparser()
name = arg['label_set_name']

try:
    os.mkdir(f"results/{name}")
except FileExistsError:
    pass

folders = [f"simulation_{k}" for k in (2, 3, 4, 5, 8)]

total_img_number = 0

IoU = {"Background": 0, "Stone": 0, "Border": 0, "Mean": 0}

for f in folders:
    # copy
    shutil.copy(f"{f}/results/{name}.npy", f"results/{name}/{f}.npy")

    # get number of images
    n_images = len(os.listdir(f'{f}/images/'))
    print(n_images)
    total_img_number += n_images

    # read results npy
    instance_results = np.load(f"results/{name}/{f}.npy", allow_pickle=True).item()
    print(f, " :")
    print(instance_results)
    for key, value in instance_results.items():
        IoU[key] = IoU[key] + value * n_images

print("Total image number: ", total_img_number)
for key, value in IoU.items():
    IoU[key] = IoU[key]/total_img_number

print("Average Results:")
print(IoU)

np.save(f'results/{name}/result.npy', IoU)
