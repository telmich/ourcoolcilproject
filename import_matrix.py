#!/usr/bin/env python3

import argparse
import sys


def insert_line_in_matrix(matrix, line):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matrix importer for CIL by Nico Schottelius, 2017')
    parser.add_argument('--infile', help='File to read data from', required=True)
    parser.add_argument('--default-value', type=float,
                        help='Value for missing entries in the matrix', required=False)
    parser.add_argument('--random-matrix', type=float,
                        help='Value for missing entries in the matrix', required=False)

    parser.add_argument('--matrix-rows', type=int,
                        help='Rows for the matrix', required=False, default=1000)
    parser.add_argument('--matrix-columns', type=int,
                        help='Columns for the matrix', required=False, default=10000)

    args = parser.parse_args()

    print(args)

    try:
        f = open(args.infile, "r")
    except Exception as e:
        print(e)
        sys.exit(1)
