"""
Node classification Task For Evaluation:
Full explanation for what is done is in the survey file in our github page.
Code explanation:
For this task, one should have a labeled graph: 2 files are required: Graph edges in '.edgelist' or '.txt' format and
nodes' labels in '.txt' format. For example labeled graphs you can enter to the link in the github page. You should
insert the in the appropriate place in the main function in the file 'directed_cheap_node2vec' or
'undirected_cheap_node2vec', depends on your graph's type.
This task compares two things:
1. Compare performance of our method and regular node2vec, meaning we do the same task with both methods, calculate
    needed scores and compare between them - This would be mission 1.
2. Only for our method, compare the success of the task (measuring by several scores) for different number of nodes
    in the initial projection - This would be mission 2.
For mission 1: Go to the file 'directed_cheap_node2vec' (or undirected). In the main function change 'initial' variable
    to a list that consists the percentage of nodes you want in the initial projection (for example, for pubmed2, 0.975
    means 100 nodes in the initial projection). Go back to this file and run main(1, 'directed'/'undirected').
For mission 2: Go to the file 'directed_cheap_node2vec' (or undirected). In the main function, change 'initial' variable
    to a list that consists a number of percentages of nodes you want in the initial projection (for example, for
    pubmed2 the list is [0.975, 0.905, 0.715, 0.447, 0.339], meaning run with 100 nodes in the initial projection, then
    with 1000, 3000, 7000 and 10000). Go back to this file to the function 'initial_proj_vs_scores' and replace x
    to be equal to 'initial' list you changed earlier. Then, you can run this file- main(2, 'directed'/'undirected'.
"""


try: import cPickle as pickle
except: import pickle
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
# from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from node2vec import Node2Vec
import matplotlib as mpl
import matplotlib.pyplot as plt


# for plots that will come later
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14


"""
Code for the node classification task as explained in GEM article. This part of the code belongs to GEM.
For more information, you can go to our github page.
"""


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction


def evaluateNodeClassification(X, Y, test_ratio):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro and accuracy.
    """
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(
        X,
        Y,
        test_size=test_ratio
    )
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr())
    classif2.fit(X_train, Y_train)
    prediction = classif2.predict(X_test, top_k_list)
    accuracy = accuracy_score(Y_test, prediction)
    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')
    return micro, macro, accuracy


def expNC(X, Y, test_ratio_arr, rounds):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average
    :return: Scores for all splits and all splits- F1-micro, F1-macro and accuracy.
    """
    micro = [None] * rounds
    macro = [None] * rounds
    acc = [None] * rounds

    for round_id in range(rounds):
        micro_round = [None] * len(test_ratio_arr)
        macro_round = [None] * len(test_ratio_arr)
        acc_round = [None] * len(test_ratio_arr)

        for i, test_ratio in enumerate(test_ratio_arr):
            micro_round[i], macro_round[i], acc_round[i]= evaluateNodeClassification(X, Y, test_ratio)

        micro[round_id] = micro_round
        macro[round_id] = macro_round
        acc[round_id] = acc_round

    micro = np.asarray(micro)
    macro = np.asarray(macro)
    acc = np.asarray(acc)

    return micro, macro, acc


def calculate_avg_score(score, rounds):
    """
    Given the lists of scores for every round of every split, calculate the average score of every split.
    :param score: F1-micro / F1-macro / Accuracy
    :param rounds: How many times the experiment has been applied for every split.
    :return: Average score for every split
    """
    all_avg_scores = []
    for i in range(score.shape[1]):
        # micro1 and macro1
        avg_score = (np.sum(score[:, i])) / rounds
        all_avg_scores.append(avg_score)
    return all_avg_scores


def calculate_all_avg_scores(micro1, macro1, micro2, macro2, acc1, acc2, rounds):
    """
    For all scores calculate the average score for every split. The function returns list for every
    score type- 1 for cheap node2vec and 2 for regular node2vec.
    """
    all_avg_micro1 = calculate_avg_score(micro1, rounds)
    all_avg_macro1 = calculate_avg_score(macro1, rounds)
    all_avg_micro2 = calculate_avg_score(micro2, rounds)
    all_avg_macro2 = calculate_avg_score(macro2, rounds)
    all_avg_acc1 = calculate_avg_score(acc1, rounds)
    all_avg_acc2 = calculate_avg_score(acc2, rounds)
    return all_avg_micro1, all_avg_macro1, all_avg_micro2, all_avg_macro2, all_avg_acc1, all_avg_acc2


def do_graph_split(avg_score1, avg_score2, test_ratio_arr, top, bottom, score, i):
    """
    Plot a graph of the score as a function of the test split value.
    :param avg_score1: list of average scores for every test ratio, 1 for cheap node2vec.
    :param avg_score2: list of average scores for every test ratio, 2 for regular node2vec.
    :param test_ratio_arr: list of the splits' values
    :param top: top limit of y axis
    :param bottom: bottom limit of y axis
    :param score: type of score (F1-micro / F1-macro / accuracy)
    :return: plot as explained above
    """
    fig = plt.figure(i)
    plt.plot(test_ratio_arr, avg_score1, '-ok', color='blue')
    plt.plot(test_ratio_arr, avg_score2, '-ok', color='red')
    plt.legend(['chep node2vec', 'regular node2vec'], loc='upper left')
    plt.ylim(bottom=bottom, top=top)
    plt.title("Pubmed2 dataset")
    plt.xlabel("test ratio")
    plt.ylabel(score)
    return fig


def split_vs_score(avg_micro1, avg_macro1, avg_micro2, avg_macro2, avg_acc1, avg_acc2, test_ratio_arr):
    """
    For every type of score plot the graph as explained above.
    """
    # you can change borders as you like
    fig1 = do_graph_split(avg_micro1, avg_micro2, test_ratio_arr, 0.4, 0.2, "micro-F1 score", 1)
    fig2 = do_graph_split(avg_macro1, avg_macro2, test_ratio_arr, 0.5, 0, "macro-F1 score", 2)
    fig3 = do_graph_split(avg_acc1, avg_acc2, test_ratio_arr, 0.6, 0, "accuracy",3)
    return fig1, fig2, fig3


def do_graph_initial(score1, score2, number_of_nodes, bottom, top, score):
    """
    Plot a graph of the score as a function of number of nodes in the initial projection.
    :param score1: list of average scores for every test ratio, 1 for cheap node2vec.
    :param score2: list of average scores for every test ratio, 2 for regular node2vec.
    :param number_of_nodes: a list of the number of nodes in initial projection, for every experiment
    :param bottom: bottom y axis limit
    :param top: top y axis limit
    :param score: Type of score
    :return: The plot as explained above
    """
    # macro graph
    plt.plot(number_of_nodes, score1, '-ok', color='blue')
    plt.plot(number_of_nodes, score2, '-ok', color='red')
    plt.legend(['chep node2vec', 'regular node2vec'], loc='upper left')
    plt.ylim(bottom=bottom, top=top)
    plt.title("Pubmed2 dataset")
    plt.xlabel("number of nodes in the initial embedding")
    plt.ylabel(score)
    plt.show()


def initial_proj_vs_scores(micro1, micro2, macro1, macro2, acc1, acc2):
    # change x according to 'initial' list in the other file as explained in the begining og this file.
    x = [100, 1000, 3000, 7000, 10000]
    do_graph_initial(micro1, micro2, x, 0, 1, "micro-F1 score")
    do_graph_initial(macro1, macro2, x, 0, 1, "macro-F1 score")
    do_graph_initial(acc1, acc2, x, 0, 1, "accuracy")


def read_labels(file_tags, dict_proj):
    """
    Read the labels file and return the labels as a matrix. Matrix is from size number of samples by number
    of labels, where C[i,j]==1 if node i has label j, else 0.
    :param file_tags: a file with labels for every node
    :return: matrix as explained above
    """
    c = np.loadtxt(file_tags, dtype=int)
    labels = {x: y for (x, y) in c}
    keys = list(dict_proj.keys())
    Y = np.zeros((len(dict_proj), 3))
    for i in range(len(keys)):
        key = int(keys[i])
        tag = labels[key]
        for j in range(3):
            if j == tag-1:
                Y[i, j] = 1
            else:
                Y[i, j] = 0
    return Y


def cheap_node2vec(dict_proj, dim):
    """
    Run cheap node2vec and make it a features matrix- matrix of size number of sample by number of embedding
    dimension, where the i_th row of X is its projection from cheap node2vec.
    :param dict_proj: A dictionary with keys==nodes in projection and values==projection
    :return: a matrix as explained above
    """
    X = np.zeros((len(dict_proj), dim))
    keys = list(dict_proj.keys())
    for i in range(len(keys)):
        X[i, :] = dict_proj[keys[i]]
    return X


def regular_node2vec(graph, dim, dict_proj):
    """
    Run regular node2vec and make it a features matrix- matrix of size number of sample by number of embedding
    dimension, where the i_th row of X is its projection from regular node2vec.
    :param graph: Our graph
    :param dim: Dimension of the embedding
    :return: a matrix as explained above
    """
    nodes = list(graph.nodes())
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    X = np.zeros((len(nodes), dim))
    keys = list(dict_proj.keys())
    for i in range(len(keys)):
        X[i, :] = np.asarray(model.wv.get_vector(keys[i]))
    return X


def main(mission, graph_type):
    """
    The main function running the 2 missions that were explained above. If you have a directed graph please initialize
    graph_type = 'directed'. If you have an undirected graph- graph_type = 'undirected'. For mission 1 insert
    mission = 1, for mission 2 insert mission = 2.
    """
    if graph_type == 'directed':
        from directed_cheap_node2vec import list_dicts, G, file
    elif graph_type == 'undirected':
        from undirected_cheap_node2vec import list_dicts, G, file
    if file is not None:
        all_micro1 = []
        all_micro2 = []
        all_macro1 = []
        all_macro2 = []
        all_acc1 = []
        all_acc2 = []
        for i in range(len(list_dicts)):
            final_dict_proj = list_dicts[i]
            Y = read_labels(file, final_dict_proj)
            X = cheap_node2vec(final_dict_proj, 10)
            if mission == 1:
                ratio_arr = [0.2, 0.4, 0.5, 0.8]
            elif mission == 2:
                ratio_arr = [0.5]
            micro1, macro1, acc1 = expNC(X, Y, ratio_arr, 3)
            X = regular_node2vec(G, 10, final_dict_proj)
            micro2, macro2, acc2 = expNC(X, Y, ratio_arr, 3)
            avg_micro1, avg_macro1, avg_micro2, avg_macro2, avg_acc1, avg_acc2 = \
                calculate_all_avg_scores(micro1, macro1, micro2, macro2, acc1, acc2, 3)
            if mission == 1:
                fig1, fig2, fig3 = split_vs_score(avg_micro1, avg_macro1, avg_micro2, avg_macro2, avg_acc1, avg_acc2, ratio_arr)
                plt.show(fig1)
            elif mission == 2:
                all_micro1.append(avg_micro1[0])
                all_micro2.append(avg_micro2[0])
                all_macro1.append(avg_macro1[0])
                all_macro2.append(avg_macro2[0])
                all_acc1.append(avg_acc1[0])
                all_acc2.append(avg_acc2[0])
        if mission == 2:
            initial_proj_vs_scores(all_micro1, all_micro2, all_macro1, all_macro2, all_acc1, all_acc2)


# here you can change for type of graph you have (directed or undirected) and mission you want
main(1, 'directed')
