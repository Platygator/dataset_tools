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
# ann_dir = 'r18_base_umarked'
full_dir = arg['test_set']
base_dir = "boulderSet/test_set_save/"
ann_dir = 'real_2_gt/masks'

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
    label_img = cv2.imread(img_name, 0)
    labels_present = np.unique(label_img)
    rand_col = np.random.random(
        [max(labels_present), 3])
    rand_col[0, :] = [0, 0, 0]
    mask_show = (rand_col[label_img.astype('uint8')] * 255).astype('uint8')
    col_label = cv2.addWeighted(img, 0.3, mask_show, 0.8, 0.0)
    visual = cv2.addWeighted(img, 0.4, col_label, 0.6, 0)
    cv2.imwrite(f'{full_dir}' + "/v_" + os.path.basename(img_name), visual)

