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
    parser.add_argument('-l', '--labels', required=False, help='label set name')
    parser.add_argument('-i', '--images', required=False, help='image set name')
    parser.add_argument('-t', '--test', required=False, help='test set name')
    args = vars(parser.parse_args())
    return args


args = create_argparser()
test_name = args["label"]

if args["labels"]:
    try:
        os.unlink("labels")
    except FileNotFoundError:
        pass
    
    os.symlink(f"{args['labels']}", "labels")

if args["images"]:
    try:
        os.unlink("images")
    except FileNotFoundError:
        pass

    os.symlink(f"{args['images']}", "images")

if args["test"]:
    try:
        os.unlink("test_set")
    except FileNotFoundError:
        pass

    os.symlink(f"{args['test']}", "test_set")


print("labels: ", os.readlink("labels"))
print("images: ", os.readlink("images"))
print("test_set: ", os.readlink("test_set"))
