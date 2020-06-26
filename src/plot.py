"""Ref
# https://github.com/bamos/densenet.pytorch/blob/master/plot.py
"""
import argparse
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


def rolling(N, i, loss, err):
    i_ = i[N - 1:]
    K = np.full(N, 1. / N)
    loss_ = np.convolve(loss, K, 'valid')
    err_ = np.convolve(err, K, 'valid')
    return i_, loss_, err_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expDir', type=str)
    args = parser.parse_args()

    try:
        trainP = os.path.join(args.expDir, 'train.csv')
        trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
        testP = os.path.join(args.expDir, 'test.csv')
        testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)
    except(IOError):
        exit()

    # N = 392 * 2  # Rolling loss over the past epoch.

    trainI, trainLoss, trainErr = np.split(trainData, [1, 2], axis=1)
    trainI, trainLoss, trainErr = [x.ravel() for x in
                                   (trainI, trainLoss, trainErr)]

    # try:
    #     trainI_, trainLoss_, trainErr_ = rolling(N, trainI, trainLoss, trainErr)
    # except(ValueError):
    #     exit()

    testI, testLoss, testErr = np.split(testData, [1, 2], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(trainI, trainLoss, label='Train')
    # plt.plot(trainI_, trainLoss_, label='Train')
    plt.plot(testI, testLoss, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    ax.set_yscale('log')
    loss_fname = os.path.join(args.expDir, 'loss.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(trainI, trainErr, label='Train')
    # plt.plot(trainI_, trainErr_, label='Train')
    plt.plot(testI, testErr, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    ax.set_yscale('log')
    plt.legend()
    err_fname = os.path.join(args.expDir, 'error.png')
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))

    loss_err_fname = os.path.join(args.expDir, 'loss-error.png')
    os.system('convert +append {} {} {}'.format(loss_fname, err_fname, loss_err_fname))
    print('Created {}'.format(loss_err_fname))
