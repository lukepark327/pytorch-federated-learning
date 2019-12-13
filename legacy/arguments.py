import argparse


def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    parser.add_argument('--nodes', metavar='N', type=int, default=100,
                        help='total number of nodes')
    parser.add_argument('--round', metavar='R', type=int, default=5000,
                        help='number of rounds')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
