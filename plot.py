import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pickle

def draw(f, t):
    df = pd.DataFrame({ 'from':f, 'to':t})

    # Build your graph. Note that we use the DiGraph function to create the graph!
    G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph() )
    # nx.draw(G, layout='tree', with_labels=True, node_size=300, alpha=1, node_color='skyblue', arrows=True, pos=nx.spectral_layout(G))
    nx.draw(G, node_size=10, alpha=0.8, node_color='skyblue', arrows=True, pos=nx.fruchterman_reingold_layout(G))

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig('dag.png', dpi=300)

if __name__ == "__main__":
    with open('dag.data', 'rb') as filehandle:
        # read the data as binary data stream
        x = pickle.load(filehandle)
        # print(x)
        draw(x["f"], x["t"])
    # draw(['D', 'A', 'B', 'C','A'], ['A', 'D', 'A', 'E','C'])
    