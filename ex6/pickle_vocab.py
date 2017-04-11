#!/usr/bin/env python3
import pickle
import sys

def help():
    print('Usage: {} output_file.pkl vocab_file.txt'.format(sys.argv[0]))

def main():
    vocab = dict()
    with open(sys.argv[2]) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(sys.argv[1], 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        help()
        sys.exit(1)
    main()
