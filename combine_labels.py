"""
Created by Jan Schiffeler on 12.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import shutil
import os
import glob
import argparse


def create_argparser():
    parser = argparse.ArgumentParser(description='Chose label name')
    parser.add_argument('-l', '--label_set_name', required=True, help='label set name')
    args = vars(parser.parse_args())
    return args


arg = create_argparser()
folders = [f"simulation_{k}" for k in (2, 3, 4, 5, 8)]

try:
    os.mkdir(f"results/{arg['label_set_name']}")
except FileExistsError:
    pass

for f in folders:
    for name in [os.path.basename(k) for k in glob.glob(f"{f}/labels/*.png")]:
        shutil.copy(f"{f}/labels/{name}", f"boulderSet/{arg['label_set_name']}/{name}")
        print(f"{f}/labels/{name}")

print(f"boulderSet/{arg['label_set_name']}")
# try:
#     os.unlink("labels")
# except FileNotFoundError:
#     pass
#
# os.symlink(f"boulderSet/{test_name}", "boulderSet/labels")
