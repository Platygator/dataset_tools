"""
Created by Jan Schiffeler on 13.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import cv2
import numpy as np
import glob
import os
import argparse


def create_argparser():
    parser = argparse.ArgumentParser(description='Chose images main_path')
    parser.add_argument('-t', '--test_set', required=False, help='path to test_set')
    args = vars(parser.parse_args())
    return args

arg = create_argparser()
base_dir = "mmseg_results/"
ann_dir = 'boulder_photos'
full_dir = arg['test_set']
# base_dir = "boulderSet/test_set"
# ann_dir = 'labels'

if not full_dir:
    full_dir = os.path.join(base_dir, ann_dir)

photo_images = [k for k in glob.glob(f'{full_dir}/*.png')]

len_img = len(photo_images)

print(os.getcwd())
for i, img_name in enumerate(photo_images):
    if os.path.basename(img_name).startswith("v_"):
        continue
    print(f"Transforming image {i+1} / {len_img} -> {img_name}")
    img = cv2.imread(f"boulderSet/test_set/images/{os.path.basename(img_name)}")
    # img = cv2.imread(f"real_2/images/{os.path.basename(img_name)}")
    # img = cv2.imread(f"boulderSet/images/{os.path.basename(img_name)}")
    label = cv2.imread(img_name, 0)
    label[label == 1] = 128
    # label[label == 127] = 128
    label[label == 2] = 255
    # label[label == 254] = 255
    label[label == 3] = 50
    visual = cv2.addWeighted(img, 0.6, label[:, :, np.newaxis].repeat(3, axis=2), 0.4, 0)
    # print("nothing :)")
    cv2.imwrite(f'{full_dir}' + "/v_" + os.path.basename(img_name), visual)

