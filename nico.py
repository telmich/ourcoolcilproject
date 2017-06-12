import random
import time

import numpy as np

# From exercise 5
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

################################################################################
# EX1
#
def matrix_standardization_v1(matrix):
    """
    subtract mean and divide by standard deviation

    Does not return anything, as the matrix itself is changed

    """
    for col in matrix.T:
        mean = np.mean(col)
        std  = np.std(col)

        # zero std -> keep as is
        if std == 0:
            std = 1

        col[...] = (col - mean) / std


def matrix_standardization_v2(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    matrix = matrix - mean
    matrix = matrix / std

    return matrix

def matrix_standardization_v3(matrix):
    return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)

def matrix_normalization(matrix, axis=0):
    return (matrix - np.mean(matrix, axis=axis))

def matrix_standardization_v4(matrix, axis=0):
    return matrix_normalization(matrix, axis=axis) / np.std(matrix, axis=axis)

def matrix_standardization(matrix):
    return matrix_standardization_v4(matrix)


def pairwise_distance(a, b):
    """ return pairwise distances between two px2 and qx2 matrices"""

    make_complex = np.array([1, 1j]).reshape(2,1)

    a_complex = a.dot(make_complex)

    # Turn into 2xq for broadcasting
    b_complex = b.dot(make_complex).T

    return np.absolute(a_complex - b_complex)


def likelihood_data_sample(X, theta1, theta2):
    """ X is matrix of Xi, columns"""

    def formula(X, mu, sigma):
        # Wrong, too lazy to type real evaluation

        vec_len = X.shape[0]

        return np.empty(vec_len)

    res1 = formula(X, *theta1)
    res2 = formula(X, *theta2)

    res = res1 < res2

    # False = 0, True = 1 => res1 > res2: False+1 = 1
    return [int(x)+1 for x in res]

################################################################################
# EX 2

def load_images_into_matrix(image_dir):
    files = [ os.path.join(image_dir, image) for image in os.listdir(image_dir) ]

    print("Loading {} images".format(len(files)))

    imgs = [Image.open(image) for image in files]

    # print("Showing image 1")
    # plt.show(imgs[0])
    # time.sleep(2)

    # Assume all images have the same size
    img0 = imgs[0]
    width, height = img0.size
    print("Image size: {} {}".format(width, height))

    # Compute input matrix X: one vector per image
    X_list = [np.ravel(imgs[i].getdata()) for i in
              range(len(files))]

    # Convert it to numpy input
    X = np.array(X_list, dtype=np.float32)

    # print(X.shape)

    return (X, width)

def covariance_matrix(matrix, axis=1):
    n = matrix.shape[axis]
    return (1/n) * (matrix.dot(matrix.T))

def ex2(image_dir):
    # 1. Build a matrix collecting all images as its columns
    matrix, width = load_images_into_matrix(image_dir)
    # 2. Normalize all images by subtracting the mean
    normalized_matrix = matrix_normalization(matrix)

    # 3. Perform PCA on the covariance matrix

    # Both should work similar - np.cov takes 1/(n-1) afaik
    # cov_mat = covariance_matrix(normalized_matrix)

    # covariance
    cov_mat = np.cov(normalized_matrix.T)
    print(cov_mat.shape)

    # eigenvalues + vectors
    # Takes *very* long
    #     eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Takes *very* long
    # SVD instead of eigenvalue
    u, s, v  = np.linalg.svd(cov_mat, full_matrices=True)

    eig_vals = s
    eig_vecs = u
    print(u.shape)

    # Check that the length of eigenvectors is almost 1
    # Just for fun.
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])


    # 4. Visualize the 5 first principal components.1 what is the interpretation of these images?
    num_eigen_vectors = 5
    used_eigen_vectors = [x[1] for x in eig_pairs[:num_eigen_vectors]]

    projection_matrix = np.stack(used_eigen_vectors, axis=1)
    print(projection_matrix.shape)

    # project matrix
    projected_matrix = matrix.dot(projection_matrix)

    print(projected_matrix.shape)

    # These are the top 5 eigenfaces (?)
    for img in projected_matrix:
        img_correct_shape = img.reshape(-1, width)
        plt.imshow(img_correct_shape)
        plt.show()

    # NEXT TODO: compress one image and show

################################################################################
# EX foo

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

    return matrix

def random_non_negative_matrix(rows, columns, max_val=255):
    random.seed()

    matrix = np.empty([rows, columns])

    for x in np.nditer(matrix, op_flags=['readwrite']):
        x[...] = random.randint(0, max_val)

    # print("random matrix: {}".format(matrix))
    return matrix




def ex5_non_negative_factorisation(U, V, X, steps):
    for i in range(steps):
        U = solve_for_all_columns(V, U, X)
        V = solve_for_all_columns(U, V, X.T)

    return (U, V)

def solve_for_all_columns(U, V, X):
    # iterating over a matrix iterates over rows
    # however we want to iterate over columns,
    # so we transpose
    for idx, col in enumerate(X):
        vj = solve_for_one_column(U, col)

        # this looks correct when trying in the shell
        V[:, idx] = vj

    V = project_non_negative(V)
    return V

# X
# ValueError: shapes (3,38) and (4080,) not aligned: 38 (dim 1) != 4080 (dim 0)

# X.T
# ValueError: shapes (3,4080) and (38,) not aligned: 4080 (dim 1) != 38 (dim 0)


def solve_for_one_column(U, xcol):
    """ highly improved version that omits information for calculation"""
    A = np.dot(U, U.T)
    b = np.dot(U, xcol)

    x = np.linalg.solve(A, b)

    return x


def ex5(image_dir):
    X, width = load_images_into_matrix(image_dir)

    rows, cols = X.shape

    U = random_non_negative_matrix(3, rows)
    V = random_non_negative_matrix(3, cols)

    U, V = ex5_non_negative_factorisation(U, V, X, 20)

    ex5_show(V, width)

def ex5_show(V, width):
    new_images_stacked = np.reshape(V, (-1, width))
    print(new_images_stacked.shape)
    fig1 = plt.figure()
    plt.show(new_images_stacked)


if __name__ == '__main__':
    image_dir = "/home/users/nico/eth/cil/lecture_cil_public/exercises/ex5/CroppedYale/CroppedYale"

    # EX2
    ex2(image_dir)

    # EX 4 test code
    # U = np.ones([3,4])
    # Zt = np.ones([4,2])

    # print(U.shape)
    # print(Zt.shape)

    # print("Res: {}".format(  partial(3, 1, 1, U, Zt)))

    # EX 5
    # ex5(image_dir)
