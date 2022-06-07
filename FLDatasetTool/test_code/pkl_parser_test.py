import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import *


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--pkl',
        help='Path to pkl file form root path'
    )
    args = argparser.parse_args()

    obj_labels = None
    with open("{}/{}".format(ROOT_PATH, args.pkl), 'rb') as pkl_file:
        obj_labels = pickle.load(pkl_file)

    for obj_label in obj_labels:
        print(obj_label)


if __name__ == "__main__":
    main()
