import os
import glob
import json
import pdb
import numpy as np
from shutil import copyfile
import argparse

#pathold = os.path.expanduser('~/azayats/data/race/RACE_data/dev/high')

def main():

	parser = argparse.ArgumentParser(description='Train a model on document-level SQuAD')
    parser.add_argument('mode', choices=["paragraph", "confidence", "shared-norm", "merge", "sigmoid"])
    parser.add_argument("name", help="Output directory")
    args = parser.parse_args()
    mode = args.mode




if __name__ == "__main__":
    main()
