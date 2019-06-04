import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.dpi'] = 150


def heatmap(arr):
    ax = sns.heatmap(
        arr,
        # linewidth=0.5
    )
    plt.show()
