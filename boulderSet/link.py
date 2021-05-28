"""
Created by Jan Schiffeler on 12.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import os
import argparse


def create_argparser():
    parser = argparse.ArgumentParser(description='Chose label name')
    parser.add_argument('-l', '--label', required=True, help='label set name')
    args = vars(parser.parse_args())
    return args


args = create_argparser()
test_name = args["label"]

try:
    os.unlink("labels")
except FileNotFoundError:
    pass

os.symlink(f"{test_name}", "labels")
