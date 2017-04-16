#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import sys

def help():
    print('Usage: {} output_file.npy cooc_file.pkl'.format(sys.argv[0]))

def main():
    print("loading cooccurrence matrix")
    with open(sys.argv[2], 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))


    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):

            # fill in your SGD code here,
            # for the update resulting from co-occurence (i,j)
            ...


    sys.exit(47)
    np.save(sys.argv[1], xs)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        help()
        sys.exit(1)
    main()
