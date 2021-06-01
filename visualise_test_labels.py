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

base_dir = "mmseg_results/result_labels"
ann_dir = 'crf_all'
# base_dir = "boulderSet/test_set"
# ann_dir = 'labels'

photo_images = [k for k in glob.glob(f'{os.path.join(base_dir, ann_dir)}/*.png')]
len_img = len(photo_images)

print(os.getcwd())
for i, img_name in enumerate(photo_images):
    print(f"Transforming image {i+1} / {len_img} -> {img_name}")
    img = cv2.imread(f"boulderSet/test_set/images/{os.path.basename(img_name)}")
    label = cv2.imread(img_name, 0)
    label[label == 1] = 128
    # label[label == 127] = 128
    label[label == 2] = 255
    # label[label == 254] = 255
    label[label == 3] = 50
    visual = cv2.addWeighted(img, 0.6, label[:, :, np.newaxis].repeat(3, axis=2), 0.4, 0)
    # print("nothing :)")
    cv2.imwrite(img_name[:-15] + "v_" + os.path.basename(img_name), visual)

