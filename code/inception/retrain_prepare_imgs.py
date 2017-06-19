#!/usr/bin/env python3

import csv
import functools
import os
import os.path
import shutil
import sys

IMG_PATH = os.getenv('IMG_PATH', '../../cosmology_aux_data_170429/')
SCORE_THRESHOLD = os.getenv('SCORE_THRESHOLD', 4.0)
CATEGORIES = ('cosmology', 'other')

@functools.lru_cache()
def csv_to_dict(csv_path):
    with open(csv_path, 'r') as f:
        csv_f = csv.reader(f)
        next(csv_f)  # skip header line
        return dict((r[0], float(r[1])) for r in csv_f)

@functools.lru_cache()
def images_dict(data_path, subset):
    label_file = os.path.join(data_path, subset+".csv")
    return csv_to_dict(label_file)

def process(path, dest, subset, threshold):
    for c in CATEGORIES: os.makedirs(os.path.join(dest, c), exist_ok=True)
    for img, score in images_dict(path, subset).items():
        src = os.path.join(path, subset, img+'.png')
        dst = os.path.join(dest, CATEGORIES[0] if score > threshold else CATEGORIES[1])
        shutil.copy(src, dst)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} path/to/destination_folder'.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    dest = sys.argv[1]
    process(IMG_PATH, dest, subset='scored',  threshold=SCORE_THRESHOLD)
    # process(IMG_PATH, dest, subset='labeled', threshold=0.5)  # labels are 0 or 1, so... :-)
