from nnabla.utils.load import load

import sys


def main():
    print(sys.argv)
    print(load([sys.argv[1]]))


if __name__ == '__main__':
    main()
