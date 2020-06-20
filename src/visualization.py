"""Ref
# https://ipython.org/ipython-doc/stable/parallel/dag_dependencies.html
# https://gist.github.com/apaszke/01aae7a0494c55af6242f06fad1f8b70
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def heatmap(data,
            log=False, annot=False,
            title=None, xlabel=None, ylabel=None,
            save=False, show=False):

    if log:
        data[data == 0] = np.finfo(float).eps
        data = np.log(data)

    ax = sns.heatmap(data, annot=annot)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if show:
        plt.show()

    if save:
        pass  # TODO

    plt.close()
