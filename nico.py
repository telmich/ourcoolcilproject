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


def ex5_load(dirname):
    image_dir = "CroppedYale/"

    files = os.listdir(image_dir)
    n = len(files)
    print("Loading " + str(n) + " images")

    imgs = [Image.open(image_dir + files[i]) for i in range(n)]
    # Assume all images have the same size
    img0 = imgs[0]width, height = img0.size

    # Compute input matrix X
    X_list = [np.ravel(imgs[i].getdata()) for i in range(n)]
    X = np.array(X_list, dtype=np.float32)
    print(X.shape)


def ex5_show(V):
    new_images_stacked = np.reshape(V, (-1, width))
    fig1 = plt.figure()
    plt.imshow(new_images_stacked)


if __name__ == '__main__':

    U = np.ones([3,4])
    Zt = np.ones([4,2])

    print(U.shape)
    print(Zt.shape)

    print("Res: {}".format(  partial(3, 1, 1, U, Zt)))
