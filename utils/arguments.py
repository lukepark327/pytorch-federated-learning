import argparse


def parser():
    parser = argparse.ArgumentParser(description="Hyperparameters")

    parser.add_argument('--nodes', metavar='N', type=int, default=21,
                        help='total number of nodes')
    parser.add_argument('--rounds', metavar='R', type=int, default=5000,
                        help='number of rounds')
    parser.add_argument('--epochs', metavar='E', type=int, default=3,
                        help='number of epochs per training node')
    parser.add_argument('--dataid', metavar='D', type=str, default='mnist',
                        help='mnist/ fashion_mnist / cifar10')
    parser.add_argument('--dist', type=str, default='uniform',
                        help='uniform / random')
    
    args = parser.parse_args()
    return args
