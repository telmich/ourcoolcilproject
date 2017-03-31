import random

import numpy as np

# From exercise 5
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image


def partial(x, row, column, U, Zt):
    """
    Result is something on the lines of:

    (-2 * X_dn * Zt[d,n] ) + (2 * U[1,d]*Z[d,1] + ... + SQUARED_ELEMENT + ...)


    """

    d, k1 = U.shape
    k2, n = Zt.shape

    if not k1 == k2:
        raise Exception("Mismatching dimensions of U and Zt")

    if row > d or column > n:
        raise Exception("Selected X is not within U*Zt")

    # derivation of -2*x UZt
    a = -2 * x * Zt[column, row]

    fx = 0
    for i in range(k1):
        print(i)
        print(U[row, i])
        print(Zt[i, column])
        fx += U[row, i] * Zt[i, column]

    # derivation of -2*x UZt
    b = 2 * Zt[column, row] * fx

    return a + b


def quadratic_cost_frobenius_J(X, U, V):
    return (1/2) * (np.linalg.norm(X - (U.T * V)) )**2


def project_non_negative(matrix):
    """Compare to CIL2017-04-Non-Negative.pdf page 30"""

    for x in np.nditer(matrix, op_flags=['readwrite']):
        x[...] = max(0, x)


def random_non_negative_matrix(rows, columns, max_val=255):
    random.seed()

    matrix = np.empty([rows, columns])

    for x in np.nditer(matrix, op_flags=['readwrite']):
        x[...] = random.randint(0, max_val)



def ex5_load(image_dir):
    files = [ os.path.join(image_dir, image) for image in os.listdir(image_dir) ]

    print("Loading {} images".format(len(files)))

    imgs = [Image.open(image) for image in files]

    plt.show(imgs[0])

    # Assume all images have the same size
    img0 = imgs[0]
    width, height = img0.size

    # Compute input matrix X: one vector per image
    X_list = [np.ravel(imgs[i].getdata()) for i in
              range(len(files))]

    # Convert it to numpy input
    X = np.array(X_list, dtype=np.float32)

    print(X.shape)

    return (X, width)

def ex5_non_negative():
    pass

def ex5_show(V, width):
    new_images_stacked = np.reshape(V, (-1, width))
    fig1 = plt.figure()
    plt.show(new_images_stacked)


def ex5(image_dir):
    X, width = ex5_load(image_dir)

    # X_still_positive = project_non_negative(X)

    # ex5_show(X, width)

if __name__ == '__main__':
    # EX 4 test code
    # U = np.ones([3,4])
    # Zt = np.ones([4,2])

    # print(U.shape)
    # print(Zt.shape)

    # print("Res: {}".format(  partial(3, 1, 1, U, Zt)))

    # EX 5
    ex5("/home/users/nico/eth/cil/lecture_cil_public/exercises/ex5/CroppedYale/CroppedYale")
