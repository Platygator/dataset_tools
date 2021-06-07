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

data_root = 'mmseg_results'
ann_dir = ''

photo_images = [k for k in glob.glob(f'{os.path.join(data_root, ann_dir)}/*.png')]
len_img = len(photo_images)

for i, img_name in enumerate(photo_images):
    print(f"Transforming image {i+1} / {len_img} -> {img_name}")
    img = cv2.imread(img_name, 0)
    img[img == 1] = 128
    img[img == 2] = 255
    img[img == 3] = 50
    cv2.imshow("debug", img)
    k = cv2.waitKey(0)
    cv2.destroyWindow("debug")
    if k == ord("q"):
        break
    if k == ord("s"):
        cv2.imwrite(img_name, img)
