import base64
import csv
import functools
import random
import os
import os.path
import sys

import numpy as np
import PIL.Image

IMG_EXT=".png"
TRAIN_RATIO       = 0.7
DOWNSAMPLED_WIDTH = 100  # pixels
DOWNSAMPLED_SIZE  = DOWNSAMPLED_WIDTH**2

def on_disk_cache(f):
    PATH='./_cache'
    SEPARATOR = object()
    os.makedirs(PATH, exist_ok=True)

    def key(args_, kwargs_):
        s = repr(args_) + '|' + repr(tuple(sorted(kwargs_.items())))
        return base64.b64encode(s.encode("utf-8")).decode("utf-8")

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        fname = os.path.join(PATH, key(args, kwargs)+'.npy')
        if os.path.isfile(fname):
            print('{}: loaded from disk'.format(f.__name__))
            return np.load(fname)
        else:
            res = f(*args, **kwargs)
            np.save(fname, res)
            return res

    return decorated

@functools.lru_cache()
def csv_to_dict(csv_path):
    with open(csv_path, 'r') as f:
        csv_f = csv.reader(f)
        next(csv_f)  # skip header line
        return dict(csv_f)

@functools.lru_cache()
def image_labels_dict(data_path, subset):
    label_file = os.path.join(data_path, subset+".csv")
    return csv_to_dict(label_file)

@functools.lru_cache()
def image_ids(data_path, subset):
    keys = list(image_labels_dict(data_path, subset).keys())
    random.shuffle(keys)
    return keys

@functools.lru_cache()
def load_image(data_path, subset, id, downsample_to=None):
    filename = os.path.join(data_path, subset, id+IMG_EXT)
    image = PIL.Image.open(filename)
    if downsample_to: image = image.resize((downsample_to, downsample_to), PIL.Image.ANTIALIAS)
    return np.array(image.getdata()).reshape(image.size[0],image.size[1])

@functools.lru_cache()
@on_disk_cache
def load_downsampled_train_test_images(data_path, subset, downsample_to=DOWNSAMPLED_WIDTH):
    """Returns (train_x, train_y, test_x, test_y) as numpy arrays. `subset` is "labeled" or "scored"."""
    img_ids = image_ids(data_path, subset)
    N = len(img_ids)

    n_train = int(TRAIN_RATIO * N)
    n_test  = N - n_train
    train_x = np.zeros((n_train, downsample_to*downsample_to))
    train_y = np.zeros(n_train)
    test_x  = np.zeros((n_test, downsample_to*downsample_to))
    test_y  = np.zeros(n_test)

    for i, img_id in enumerate(img_ids):
        print("\rLoading images: {}/{}".format(i+1, N), file=sys.stderr, end='')
        (x, y, idx) = (train_x, train_y, i) if i < n_train else (test_x, test_y, i - n_train)
        x[idx] = load_image(data_path, subset, img_id, downsample_to=downsample_to).reshape(downsample_to*downsample_to).astype(np.float32)
        y[idx] = float(image_labels_dict(data_path, subset)[img_id])
    print()

    return (train_x, train_y, test_x, test_y)
