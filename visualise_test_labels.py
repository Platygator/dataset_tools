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
# base_dir = "mmseg_results/"
# ann_dir = 'snake_r50'
full_dir = arg['test_set']
base_dir = "boulderSet/test_set_save/"
ann_dir = 'real_2_gt/labels'

if not full_dir:
    full_dir = os.path.join(base_dir, ann_dir)

photo_images = [k for k in glob.glob(f'{full_dir}/*.png')]

len_img = len(photo_images)

print(os.getcwd())
for i, img_name in enumerate(photo_images):
    if os.path.basename(img_name).startswith("v_"):
        continue
    print(f"Transforming image {i+1} / {len_img} -> {img_name}")
    img = cv2.imread(f"boulderSet/test_set_save/real_2_gt/images/{os.path.basename(img_name)}")
    # img = cv2.imread(f"boulderSet/test_set_save/snake_photos/images/{os.path.basename(img_name)}")
    # img = cv2.imread(f"boulderSet/images/{os.path.basename(img_name)}")
    # img = cv2.imread(f"simulation_9/images/{os.path.basename(img_name)}")
    # img = cv2.imread(f"boulderSet/images/{os.path.basename(img_name)}")
    label = cv2.imread(img_name, 0)
    if label is None:
        print("[ERROR] With label: ", img_name)
        continue
    col_label = np.zeros([label.shape[0], label.shape[1], 3], dtype='uint8')
    col_label[label == 1] = [65, 189, 245]
    # label[label == 127] = 128
    col_label[label == 2] = [60, 90, 255]
    # label[label == 254] = 255
    col_label[label == 3] = [0, 0, 0]
    # print(np.unique(label, return_counts=True))
    visual = cv2.addWeighted(img, 0.7, col_label, 0.4, 0)
    # print("nothing :)")
    cv2.imwrite(f'{full_dir}' + "/v_" + os.path.basename(img_name), visual)