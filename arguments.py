import argparse


def parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    parser.add_argument('--nodes', metavar='N', type=int, default=21,
                        help='total number of nodes')
    parser.add_argument('--overlap', metavar='O', type=float, default=1.,
                        help='portion of overlapping data (allow no overlap: 1.0)')
    parser.add_argument('--dist', metavar='D', type=str, default="uniform",
                        help='mechanism of total data distribution')  # uniform, normal, pareto, et al.
    parser.add_argument('--bias', metavar='B', type=str, default="uniform",
                        help='mechanism of one\'s data distribution')  # uniform, normal, pareto, just one label, et al.

    parser.add_argument('--round', metavar='R', type=int, default=5000,
                        help='TBA')
    parser.add_argument('--top', metavar='T', type=int, default=3,
                        help='TBA')
    parser.add_argument('--dataset', metavar='S', type=int, default=21000,
                        help='TBA')

    args = parser.parse_args()
    # print(args.nodes)
    return args


# main
if __name__ == "__main__":
    parser()
