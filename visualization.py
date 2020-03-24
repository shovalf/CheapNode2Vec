"""
Visualization Task for Evaluation. For full details please enter to the survey file in git.
Code explanation:
For this task, one should have a labeled graph: 2 files are required: Graph edges in '.edgelist' or '.txt' format and
nodes' labels in '.txt' format. For example labeled graphs you can enter to the link in the github page. You should
insert the in the appropriate place in the main function in the file 'directed_cheap_node2vec' or
'undirected_cheap_node2vec' - depends on your graph.
Go to the file 'directed_cheap_node2vec' (or undirected). In the main function change 'initial' variable to a list
that consists the percentage of nodes you want in the initial projection (for example, for pubmed2, 0.975 means 100
nodes in the initial projection). Go back to this file and run main('directed') or main('undirected'.
"""

import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib as mpl


mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 20


def regular_node2vec(G, nodes):
    """
    Run regular node2vec
    :param G: Our graph
    :param nodes: nodes of the graph
    :return: a list of projections of every node as np arrays
    """
    node2vec = Node2Vec(G, dimensions=10, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    projections = []
    for i in range(len(nodes)):
        projections.append(np.asarray(model.wv.get_vector(nodes[i])))
    return projections


def cheap_node2vec(final_dict_proj):
    """
    Run cheap node2vec
    :param final_dict_proj: A dictionary with keys==nodes in projection and values==projection
    :return: a list of projections of every node as np arrays
    """
    keys = list(final_dict_proj.keys())
    projections = []
    for i in range(len(keys)):
        projections.append(final_dict_proj[keys[i]])
    return projections


def read_labels(file):
    """
    Read the labels file and return the labels as a list
    :param file: a file with labels for every node
    :return: a list of the labels of every node
    """
    c = np.loadtxt(file, dtype=int)
    labels = {x: y for (x, y) in c}
    list_labels = []
    keys = list(labels.keys())
    for i in range(len(keys)):
        list_labels.append(labels[keys[i]])
    return list_labels


def visualization(projections, nodes, i, labels):
    """
    The visualization task explained in details in the pdf file attached in the github.
    :param projections: a list of projections of every node as np arrays
    :param nodes: nodes of the graph
    :param i: number of figure
    :param labels:a list of the labels of every node
    :return: tsne representation of both regular node2vec and cheap node2vec
    """
    projections = np.asarray(projections)

    names = []
    for i in range(len(nodes)):
        names.append(nodes[i])

    data = pd.DataFrame(projections)
    X = data.values

    tsne_points = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(i)
    ax = fig.add_subplot(111)
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')

    if len(labels) == 0:
        ax.scatter(tsne_points[:, 0], tsne_points[:, 1], cmap='tab10', alpha=0.8, s=8)
        plt.title('Regular Node2vec Representation with TSNE')
    else:
        ax.scatter(tsne_points[:, 0], tsne_points[:, 1], labels, cmap='tab10', alpha=0.8, s=8)
        plt.title('Cheap Node2vec Representation with TSNE')
    plt.show()


def main(mission):
    """
    Main function to run all task. It can be run on both directed and undirected graph, determined
    by the mission variable- 'directed' or 'undirected'
    :param mission: 'directed' or 'undirected'
    :return: tsne representation of both regular node2vec and cheap node2vec
    """
    if mission == 'directed':
        from directed_cheap_node2vec import list_dicts, G, file
        final_dict_proj = list_dicts[0]
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            nodes[i] = str(nodes[i])
        if file is not None:
            labels = read_labels(file)
        else:
            labels = []
        projections_regular = regular_node2vec(G, nodes)
        projections_cheap = cheap_node2vec(final_dict_proj)
        visualization(projections_regular, nodes, 1, labels)
        visualization(projections_cheap, nodes, 2, labels)
    elif mission == 'undirected':
        from undirected_cheap_node2vec import list_dicts, G, file
        final_dict_proj = list_dicts[0]
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            nodes[i] = str(nodes[i])
        if file is not None:
            labels = read_labels(file)
        else:
            labels = []
        projections_regular = regular_node2vec(G, nodes)
        projections_cheap = cheap_node2vec(final_dict_proj)
        visualization(projections_regular, nodes, 1, labels)
        visualization(projections_cheap, nodes, 2, labels)
    else:
        print("If your graph is directed please input 'directed', for an undirected graph"
              "please input 'undirected'")


# can be changed to undirected, depends on the graph
main('directed')
