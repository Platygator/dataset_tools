"""
Created by Jan Schiffeler on 01.06.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:

cam_mat = np.array([[1577.1159987660135, 0, 676.7292997380368],
                    [0, 1575.223362703865, 512.8101184300463],
                    [0, 0, 1]])

dist_mat = np.array([-0.46465317710098897, 0.2987490394355827, 0.004075959465516531, 0.005311175696501367])

cam_mat = np.array([[1061.029930829013, 0, 1011.539146901374],
                    [0, 1061.01597110039, 762.8834144400216],
                    [0, 0, 1]])

dist_mat = np.array([-0.007917994945883303, 0.03351829372987945, 0.0003841044102097604, -0.002080542086396206, 0])
"""

import numpy as np
import os
import glob
import argparse
import cv2


def create_argparser():
    parser = argparse.ArgumentParser(description='Chose images main_path')
    parser.add_argument('-p', '--main_path', required=True, help='main_path to images')
    parser.add_argument('-f', '--prefix', required=False, default='', help='main_path to images')
    parser.add_argument('-c', '--clahe', action='store_true', help='do clahe histogram normalization')
    parser.add_argument('-m', '--mask', action='store_true', help='create mask')
    args = vars(parser.parse_args())
    return args


# Change parameters here!
cam_mat = np.array([[1061.029930829013, 0, 1011.539146901374],
                    [0, 1061.01597110039, 762.8834144400216],
                    [0, 0, 1]])

dist_mat = np.array([-0.007917994945883303, 0.03351829372987945, 0.0003841044102097604, -0.002080542086396206, 0])

# new_size = (2064//2, 1544//2)
new_size = (752, 480)
# new_size = (1440, 1080)

arg = create_argparser()
main_path = arg['main_path']
clahe = arg['clahe']
prefix = arg['prefix']
gen_mask = arg['mask']

photo_images = [k for k in glob.glob(f'{os.path.abspath(main_path)}/*.png')]
len_img = len(photo_images)

img = cv2.imread(photo_images[0])
height, width = img.shape[:2]


# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_mat, (width, height), 1, (width, height))

m1, m2 = cv2.initUndistortRectifyMap(cameraMatrix=cam_mat, distCoeffs=dist_mat, newCameraMatrix=cam_mat,
                                     R=np.eye(3), size=(width, height), m1type=cv2.CV_32FC1)
scaling_ratio = (new_size[0]/width, new_size[1]/height)

edge = 5
mask = np.ones([height - edge, width - edge])
mask = np.pad(mask, [[edge, edge], [edge, edge]], constant_values=0)
mask = cv2.remap(mask, map1=m1, map2=m2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                 borderValue=(0, 0, 0, 0))
mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_AREA)
cv2.imwrite(f"{main_path}/mask.png", mask * 255)

for i, path in enumerate(photo_images):
    print(f"[INFO] Undistorting image {i + 1}/{len_img} <- {path}")
    img = cv2.imread(path)

    # Undistortion
    # img = cv2.undistort(img, cam_mat, dist_mat, None, newcameramtx)
    # x, y, w, h = roi
    # img = img[y:y + h, x:x + w]
    img = cv2.remap(img, map1=m1, map2=m2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))

    # Apply histogram normalization
    if clahe:
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        rgb_planes = cv2.split(img)
        cl_img = []
        for p in rgb_planes:
            cl_img.append(cl.apply(p))

        img = cv2.merge(cl_img)

    # Resizing
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Saving
    path_parts = list(os.path.split(path))
    img_name = prefix + path_parts[-1]
    path_parts.pop(-1)
    cv2.imwrite(os.path.join(*path_parts, img_name), img)
    if gen_mask:
        try:
            os.symlink(f"{main_path}/mask.png", f"{os.path.join(*path_parts, img_name)}.png")
        except FileExistsError:
            pass


scaling_mat = np.array([[scaling_ratio[0], 0, 0],
                       [0, scaling_ratio[1], 0],
                       [0, 0, 1]])
resized_cam_mat = scaling_mat.dot(cam_mat)

print("[INFO] Applied scaling ration: ", scaling_ratio)
print("[INFO] New camera matrix: ")
print(resized_cam_mat)
