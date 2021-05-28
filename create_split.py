"""
Created by Jan Schiffeler on 12.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import os
import glob
from numpy import random as npr

data_root = 'boulderSet'
img_dir = 'images'
test_dir = 'test_set/images'
split_dir = 'splits'

p_train = 7/10
p_val = 3/10

try:
    os.mkdir(os.path.join(data_root, split_dir))
except FileExistsError:
    pass

photo_images = [os.path.basename(k)[:-4] for k in glob.glob(f'{os.path.join(data_root, img_dir)}/*.png')]
test_images = [os.path.basename(k)[:-4] for k in glob.glob(f'{os.path.join(data_root, test_dir)}/*.png')]
npr.shuffle(photo_images)
train_length = int(len(photo_images) * p_train)
val_length = int(len(photo_images) * p_val) + train_length
print("Found ", len(photo_images), " images.")

# Create text files
with open(os.path.join(data_root, split_dir, 'train.txt'), 'w') as f:
    f.writelines(line + '\n' for line in photo_images[:train_length])

with open(os.path.join(data_root, split_dir, 'val.txt'), 'w') as f:
    f.writelines(line + '\n' for line in photo_images[train_length:val_length])

with open(os.path.join(data_root, split_dir, 'test.txt'), 'w') as f:
    f.writelines(line + '\n' for line in test_images)