#!/usr/bin/env python3

import argparse
import re
import sys
import numpy as np


def lines_to_matrix(matrix, lines):
    for line in lines:
        parsed_line = re.search("r(?P<row>[0-9]*)_c(?P<column>[0-9]*),(?P<value>.*)", line)

        # FIXME: correct indeces on out data
        row = int(parsed_line.group('row')) - 1
        column = int(parsed_line.group('column')) -1
        value = parsed_line.group('value')

        # We actually get a transposed
        matrix[row, column] = value

    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matrix importer for CIL by Nico Schottelius, 2017')
    parser.add_argument('--infile', help='File to read data from', required=True)

    parser.add_argument('--default-value', type=float,
                        help='Value for missing entries in the matrix', required=False)
    parser.add_argument('--random-matrix', action='store_true',
                        help='Create matrix with random values', required=False)

    parser.add_argument('--matrix-rows', type=int,
                        help='Rows for the matrix', required=False, default=10000)
    parser.add_argument('--matrix-columns', type=int,
                        help='Columns for the matrix', required=False, default=1000)

    args = parser.parse_args()

    # print(args)

    if args.random_matrix and args.default_value:
        print("Random matrix OR select default value, not both")
        sys.exit(1)

    if args.random_matrix == None and args.default_value == None:
        print("Specify either random matrix OR select a default value")
        sys.exit(1)

    shape = [args.matrix_rows, args.matrix_columns]

    if not args.random_matrix == None:
        the_matrix = np.empty(shape)
    if not args.default_value == None:
        the_matrix = np.ones(shape) * args.default_value


    try:
        f = open(args.infile, "r")
        lines = f.readlines()
        lines = lines[2:] # Skip header
        lines = [line.rstrip('\n') for line in lines] # Remove silly \n

    except Exception as e:
        print(e)
        sys.exit(1)

    the_matrix = lines_to_matrix(the_matrix, lines)

    print(the_matrix)
