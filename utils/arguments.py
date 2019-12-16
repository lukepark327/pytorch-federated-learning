import argparse


def parser():
    parser = argparse.ArgumentParser(description="Hyperparameters")

    parser.add_argument('--nodes', metavar='N', type=int, default=21,
                        help='total number of nodes')
    parser.add_argument('--rounds', metavar='R', type=int, default=20,
                        help='number of rounds')
    parser.add_argument('--epochs', metavar='E', type=int, default=1,
                        help='number of epochs per training node')
    parser.add_argument('--dataid', metavar='D', type=int, default=0,
                        help='mnist/ fashion_mnist / cifar10')
    parser.add_argument('--dist', type=int, default=0,
                        help='uniform / random')
    parser.add_argument('--evalrate', type=float, default=0.3,
                        help="nodes' evaluation frequency 0~1")
    parser.add_argument('--updaterate', type=float, default=0.3,
                        help="nodes' update frequency 0~1")
    
    args = parser.parse_args()
    return args
