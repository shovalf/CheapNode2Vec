import numpy as np
import networkx as nx
from node2vec import Node2Vec
import time
import heapq
from sklearn.linear_model import LinearRegression
import time


def user_print(item, user_wish):
    """
    A function to show the user the state of the code. If you want a live update of the current state of the code and
    some details: set user wish to True else False
    """
    if user_wish is True:
        print(item, sep=' ', end='', flush=True)
        time.sleep(3)
        print(" ", end='\r')


def get_initial_proj_nodes(G, key):
    """
    Function that gets the graph and return the nodes that we would like them to be in the initial projection.
    :param G: Our graph
    :param key: Controls number of nodes in the initial projection
    :return: A list of the nodes that are in the initial projection
    """
    # a dictionary of the nodes and their degrees
    dict_degrees = dict(G.degree(G.nodes()))
    # a dictionary of the nodes and the average degrees
    dict_avg_neighbor_deg = nx.average_neighbor_degree(G)
    # sort the dictionary
    sort_degrees = sorted(dict_degrees.items(), key=lambda pw: (pw[1], pw[0]))  # list
    sort_avg_n_d = sorted(dict_avg_neighbor_deg.items(), key=lambda pw: (pw[1], pw[0]))  # list
    # choose only some percents of the nodes with the maximum degree / average degree
    top_deg = sort_degrees[int(key * len(sort_degrees)):len(sort_degrees)]
    top_avgn_deg = sort_avg_n_d[int(key * len(sort_avg_n_d)):len(sort_avg_n_d)]
    # choose the nodes that have maximum degree and also maximum average degree
    tmp_deg = top_deg
    tmp_n_deg = top_avgn_deg
    for i in range(len(top_deg)):
        tmp_deg[i] = list(tmp_deg[i])
        tmp_deg[i][1] = 5
    for i in range(len(top_avgn_deg)):
        tmp_n_deg[i] = list(tmp_n_deg[i])
        tmp_n_deg[i][1] = 10
    # the nodes with the maximal degree- the nodes we want to do the projection on
    final_nodes = np.intersect1d(tmp_n_deg, tmp_deg)
    list_final_nodes = list(final_nodes)
    for i in range(len(list_final_nodes)):
        list_final_nodes[i] = str(list_final_nodes[i])
    return list_final_nodes


def create_sub_G(proj_nodes, G):
    """
    Creating a new graph from the nodes in the initial projection so we can do the node2vec projection on it
    :param proj_nodes: The nodes in the initial projection
    :param G: Our graph
    :return: A sub graph of G that its nodes are the nodes in the initial projection.
    """
    sub_G = G.subgraph(list(proj_nodes))
    return sub_G


def our_node2vec(H, dim):
    """
    A function to do node2vec embedding on the sub graph H we fedined in the function above
    :param H: Our sub graph
    :param dim: The dimension after performing node2vec
    :return: A dictionary of projections when value==node and key==projection.
    """
    # perform regular node2vec on the sub graph H
    node2vec = Node2Vec(H, dimensions=dim, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    nodes = list(H.nodes())
    # create the dictionary
    projections = {}
    for i in range(len(nodes)):
        projections.update({nodes[i]: np.asarray(model.wv.get_vector(nodes[i]))})
    return projections


def create_dict_neighbors(G):
    """
    Create a dictionary of neighbors.
    :param G: Our graph
    :return: neighbors_dict when value==node and key==set_of_neighbors (both incoming and outgoing)
    """
    G_nodes = list(G.nodes())
    neighbors_dict = {}
    for i in range(len(G_nodes)):
        node = G_nodes[i]
        neighbors_dict.update({node: set(G[node])})
    return neighbors_dict


def create_dicts_same_nodes(my_set, neighbors_dict, node, dict_out, dict_in):
    """
    A function to create useful dictionaries to represent connection between nodes that have the same type, i.e between
    nodes that are in the projection and between nodes that aren't in the projection. It depends on the input.
    :param my_set: Set of the nodes that aren't in the projection OR Set of the nodes that are in the projection
    :param neighbors_dict: Dictionary of all nodes and neighbors (both incoming and outgoing)
    :param node: Current node we're dealing with
    :param dict_out: explained below
    :param dict_in: explained below
    :return: There are 4 possibilities (2 versions, 2 to every version):
            A) 1. dict_node_node_out: key == nodes not in projection , value == set of outgoing nodes not in projection
                 (i.e there is a directed edge (i,j) when i is the key node and j isn't in the projection)
               2. dict_node_node_in: key == nodes not in projection , value == set of incoming nodes not in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j isn't in the projection)
            B) 1. dict_enode_enode_out: key == nodes in projection , value == set of outgoing nodes in projection
                 (i.e there is a directed edge (i,j) when i is the key node and j is in the projection)
               2. dict_enode_enode_in: key == nodes in projection , value == set of incoming nodes in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the projection)
    """
    set1 = neighbors_dict[node].intersection(my_set)
    if (len(set1)) > 0:
        dict_out.update({node: set1})
        neigh = list(set1)
        for j in range(len(neigh)):
            if dict_in.get(neigh[j]) is None:
                dict_in.update({neigh[j]: set([node])})
            else:
                dict_in[neigh[j]].update(set([node]))
    return dict_out, dict_in


def create_dict_node_enode(set_proj_nodes, neighbors_dict, H, node, dict_node_enode, dict_enode_node):
    """
    A function to create useful dictionaries to represent connections between nodes that are in the projection and
    nodes that are not in the projection.
    :param set_proj_nodes: Set of the nodes that are in the projection
    :param neighbors_dict: Dictionary of all nodes and neighbors (both incoming and outgoing)
    :param H:  H is the undirected version of our graph
    :param node: Current node we're dealing with
    :param dict_node_enode: explained below
    :param dict_enode_node: explained below
    :return: 1. dict_node_enode: key == nodes not in projection, value == set of outdoing nodes in projection (i.e
                    there is a directed edge (i,j) when i is the key node and j is in the projection)
             2. dict_enode_node: key == nodes not in projection, value == set of incoming nodes in projection (i.e
                    there is a directed edge (j,i) when i is the key node and j is in the projection)
    """
    set2 = neighbors_dict[node].intersection(set_proj_nodes)
    set_all = set(H[node]).intersection(set_proj_nodes)
    set_in = set_all - set2
    if len(set2) > 0:
        dict_node_enode.update({node: set2})
    if len(set_in) > 0:
        dict_enode_node.update({node: set_in})
    return dict_node_enode, dict_enode_node


def create_dicts_of_connections(set_proj_nodes, set_no_proj_nodes, neighbors_dict, G):
    """
     A function that creates 6 dictionaries of connections between different types of nodes.
    :param set_proj_nodes: Set of the nodes that are in the projection
    :param set_no_proj_nodes: Set of the nodes that aren't in the projection
    :param neighbors_dict: Dictionary of neighbours
    :return: 6 dictionaries, explained above (in the two former functions)
    """
    dict_node_node_out = {}
    dict_node_node_in = {}
    dict_node_enode = {}
    dict_enode_node = {}
    dict_enode_enode_out = {}
    dict_enode_enode_in = {}
    list_no_proj = list(set_no_proj_nodes)
    list_proj = list(set_proj_nodes)
    H = G.to_undirected()
    for i in range(len(list_no_proj)):
        node = list_no_proj[i]
        dict_node_node_out, dict_node_node_in = create_dicts_same_nodes(set_no_proj_nodes, neighbors_dict, node,
                                                                        dict_node_node_out, dict_node_node_in)
        dict_node_enode, dict_enode_node = create_dict_node_enode(set_proj_nodes, neighbors_dict, H, node,
                                                                  dict_node_enode, dict_enode_node)
    for i in range(len(list_proj)):
        node = list_proj[i]
        dict_enode_enode_out, dict_enode_enode_in = create_dicts_same_nodes(set_proj_nodes, neighbors_dict, node,
                                                                        dict_enode_enode_out, dict_enode_enode_in)
    return dict_node_node_out, dict_node_node_in, dict_node_enode, dict_enode_node, dict_enode_enode_out, dict_enode_enode_in


def calculate_average_projection_second_order(dict_proj, node, dict_enode_enode, average_two_order_proj, dim):
    """
    A function to calculate the average projections of the second order neighbors, both outgoing and incoming,
    depends on the input.
    :param dict_proj: Dict of projections (key==node, value==projection)
    :param node: Current node we're dealing with
    :param dict_enode_enode: key==node in projection , value==set of neighbors that are in the projection. Direction
            (i.e outgoing or incoming depends on the input)
    :param average_two_order_proj: explained below
    :return: Average projection of second order neighbours, outgoing or incoming.
    """
    two_order_neighs = dict_enode_enode.get(node)
    k2 = 0
    # if the neighbors in the projection also have neighbors in the projection calculate the average projection
    if two_order_neighs is not None:
        two_order_neighs_in = list(two_order_neighs)
        k2 += len(two_order_neighs_in)
        two_order_projs = []
        for i in range(len(two_order_neighs_in)):
            two_order_proj = dict_proj[two_order_neighs_in[i]]
            two_order_projs.append(two_order_proj)
        two_order_projs = np.array(two_order_projs)
        two_order_projs = np.mean(two_order_projs, axis=0)
    # else, the average projection is 0
    else:
        two_order_projs = np.zeros(dim)
    average_two_order_proj.append(two_order_projs)
    return average_two_order_proj


def calculate_projection_of_neighbors(proj_nodes, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim):
    """
    A function to calculate average degree of first order neighbors and second order neighbors, direction
    (outgoing or incoming)depends on the input.
    :param proj_nodes: Neighbors that are in the projection, direction depends on the input.
    :param dict_proj: Dict of projections (key==node, value==projection)
    :return: Average degree of first order neighbors and second order neighbors
    """
    proj = []
    # average projections of the two order neighbors, both incoming and outgoing
    average_two_order_proj_in = []
    average_two_order_proj_out = []
    list_proj_nodes = list(proj_nodes)
    # the number of first order neighbors
    k1 = len(proj_nodes)
    # to calculate the number of the second order neighbors
    k2 = 0
    # to calculate the average projection of the second order neighbors
    for k in range(len(list_proj_nodes)):
        node = list_proj_nodes[k]
        average_two_order_proj_in = calculate_average_projection_second_order(dict_proj, node, dict_enode_enode_in,
                                                                              average_two_order_proj_in, dim)
        average_two_order_proj_out = calculate_average_projection_second_order(dict_proj, node, dict_enode_enode_out,
                                                                               average_two_order_proj_out, dim)
        proj.append(dict_proj[node])
    # for every neighbor we have the average projection of its neighbors, so now do average on all of them
    average_two_order_proj_in = np.array(average_two_order_proj_in)
    average_two_order_proj_in = np.mean(average_two_order_proj_in, axis=0)
    average_two_order_proj_out = np.array(average_two_order_proj_out)
    average_two_order_proj_out = np.mean(average_two_order_proj_out, axis=0)
    proj = np.array(proj)
    # find the average proj
    proj = np.mean(proj, axis=0)
    return proj, average_two_order_proj_in, average_two_order_proj_out, k1, k2


def calculate_projection(proj_nodes_in, proj_nodes_out, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim, alpha1, alpha2, beta_11, beta_12, beta_21, beta_22):
    """
    A function to calculate the final projection of the node by a formula.
    Notice the formula and definitions of x1,x2,z11,z12,z21,z22 are in the github page.
    :param proj_nodes_in: projections of first order incoming neighbors.
    :param proj_nodes_out: projections of first order outgoing neighbors.
    :param dict_proj: Dict of projections (key==node, value==projection)
    :param dim: Dimension of the projection
    :param alpha1, alpha2, beta_11, beta_12, beta_21, beta_22: Parameters to calculate the final projection, explained
    in the github page.
    :return: The final projection of our node.
    """
    if len(proj_nodes_in) > 0:
        x_1, z_11, z_12, k1, k2 = calculate_projection_of_neighbors(
            proj_nodes_in, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim)
    else:
        x_1, z_11, z_12 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    if len(proj_nodes_out) > 0:
        x_2, z_21, z_22, k1, k2 = calculate_projection_of_neighbors(
            proj_nodes_out, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim)
    else:
        x_2, z_21, z_22 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    # the final projection of the node
    final_proj = alpha1*x_1+alpha2*x_2 - beta_11*z_11 - beta_12*z_12 - \
                 beta_21*z_21 - beta_22*z_22
    return final_proj


def first_changes(dict_1, dict_2, dict_3, node):
    """
    Technical changes need to be done after adding the node to the projection
    :param dict_1: dict_node_enode OR dict_enode_node
    :param dict_2: dict_enode_enode_out OR dict_enode_enode_in
    :param dict_3: dict_enode_enode_in OR dict_enode_enode_out
    :param node: Current node
    :return: Dicts after changes
    """
    if dict_1.get(node) is not None:
        enode = dict_1[node]
        dict_1.pop(node)
        dict_2.update({node: enode})
        enode = list(enode)
        for i in range(len(enode)):
            out_i = enode[i]
            if dict_3.get(out_i) is not None:
                dict_3[out_i].update(set([node]))
            else:
                dict_3.update({out_i: set([node])})
    return dict_1, dict_2, dict_3


def second_changes(dict_1, dict_2, dict_3, node):
    """
    Technical changes need to be done after adding the node to the projection
    :param dict_1: dict_node_node_out OR dict_node_node_in
    :param dict_2: dict_node_node_in OR dict_node_node_out
    :param dict_3: dict_enode_node OR dict_node_enode
    :param node: Current node
    :return: Dicts after changes
    """
    if dict_1.get(node) is not None:
        relevant_n_e = dict_1[node]
        dict_1.pop(node)
        if len(relevant_n_e) > 0:
            # loop of non embd neighbors
            relevant_n_e1 = list(relevant_n_e)
            for j in range(len(relevant_n_e)):
                tmp_append_n_n = dict_2.get(relevant_n_e1[j])
                if tmp_append_n_n is not None:
                    # if relevant_n_e1[j] in dict_node_node:
                    tmp_append_n_n = tmp_append_n_n-set([node])
                    dict_2[relevant_n_e1[j]] = tmp_append_n_n
                tmp_append = dict_3.get(relevant_n_e1[j])
                if tmp_append is not None:
                    # add our node to the set cause now our node is in embd
                    tmp_append.update(set([node]))
                    dict_3[relevant_n_e1[j]] = tmp_append
                else:
                    dict_3.update({relevant_n_e1[j]: set([node])})
    return dict_1, dict_2, dict_3


def one_iteration(params, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_in, dict_node_node_out,
                  dict_enode_enode_in, dict_enode_enode_out, set_n_e, current_node, dim):
    """
    One iteration of the final function. We calculate the projection and do necessary changes.
    Notice: All paranms dicts are explained above
    :param params: Best parameters to calculate a node's projection, explained in the git.
    :param set_n_e: Set of nodes that aren't in the projection
    :param current_node: The node we're dealing with at the moment.
    :param dim: The dimension of the projection
    :return: The dicts and the set of nodes not in projection because they are changed. Also return condition to tell
    if we still need to do iterations.
    """
    condition = 1
    if dict_node_enode.get(current_node) is not None:
        embd_neigh_out = dict_node_enode[current_node]
    else:
        embd_neigh_out = set()
    if dict_enode_node.get(current_node) is not None:
        embd_neigh_in = dict_enode_node[current_node]
    else:
        embd_neigh_in = set()
    # the final projection of the node
    final_proj = calculate_projection(embd_neigh_in, embd_neigh_out, dict_enode_proj, dict_enode_enode_in, dict_enode_enode_out, dim,
                                      alpha1=params[0], alpha2=params[1], beta_11=params[2],
                                      beta_12=params[3], beta_21=params[4], beta_22=params[5])

    dict_enode_proj.update({current_node: final_proj})

    # do first changes
    dict_node_enode, dict_enode_enode_out, dict_enode_enode_in = first_changes(
        dict_node_enode, dict_enode_enode_out, dict_enode_enode_in, current_node)
    dict_enode_node, dict_enode_enode_in, dict_enode_enode_out = first_changes(
        dict_enode_node, dict_enode_enode_in, dict_enode_enode_out, current_node)

    # do second changes
    dict_node_node_out, dict_node_node_in, dict_enode_node = second_changes(
        dict_node_node_out, dict_node_node_in, dict_enode_node, current_node)
    dict_node_node_in, dict_node_node_out, dict_node_enode = second_changes(
        dict_node_node_in, dict_node_node_out, dict_node_enode, current_node)

    # remove the node from the set of nodes that aren't in the projection
    set_n_e.remove(current_node)

    return condition, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_out, dict_node_node_in,\
           dict_enode_enode_out, dict_enode_enode_in, set_n_e


def final_function(pre_params, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_out, dict_node_node_in,
                   dict_enode_enode_out, dict_enode_enode_in, set_n_e, batch_precent, dim):
    """
    The final function that iteratively divided the dictionary of nodes without embedding into number of batches
    determined by batch_precent. It does by building a heap every iteration so that we enter the nodes to the
    projection from the nodes which have the most neighbors in the embedding to the least. This way the projection
    gets more accurate.
    """
    condition = 1
    k = 0
    set_n_e2 = set_n_e.copy()
    while condition > 0:
        condition = 0
        k += 1
        print(k)
        batch_size = int(batch_precent * len(set_n_e2))
        if batch_size > len(set_n_e):
            num_times = len(set_n_e)
        else:
            num_times = batch_size
        list_n_e = list(set_n_e)
        heap = []
        dict_node_enode_all = dict_node_enode.copy()
        keys = list(dict_enode_node.keys())
        for key in keys:
            if dict_node_enode.get(key) is None:
                dict_node_enode_all.update({key: dict_enode_node[key]})
            else:
                dict_node_enode_all[key].update(dict_enode_node[key])
        for i in range(len(list_n_e)):
            my_node = list_n_e[i]
            a = dict_node_enode_all.get(my_node)
            if a is not None:
                num_neighbors = len(dict_node_enode_all[my_node])
            else:
                num_neighbors = 0
            heapq.heappush(heap, [-num_neighbors, my_node])
        for i in range(len(set_n_e))[:num_times]:
            # look on node number i in the loop
            current_node = heapq.heappop(heap)[1]
            if dict_node_enode_all.get(current_node) is not None:
                condition, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_out, dict_node_node_in, \
                dict_enode_enode_out, dict_enode_enode_in, set_n_e = \
                    one_iteration(pre_params, dict_enode_proj, dict_node_enode, dict_enode_node,
                                  dict_node_node_in, dict_node_node_out,
                  dict_enode_enode_in, dict_enode_enode_out, set_n_e, current_node, dim)
    return dict_enode_proj, set_n_e


def crate_data_for_linear_regression(initial_proj_nodes, dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim):
    """
    In order to find the best parameters to calculate a node's vector, we perform linear regression. In this function
    we prepare the data for this.
    :param initial_proj_nodes: The nodes that are in the initial projection
    :param dict_intial_projections: Dictionary of nodes and their projection
    :param dict_enode_enode_in: key == nodes in projection , value == set of incoming nodes in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the projection)
    :param dict_enode_enode_out: key == nodes in projection , value == set of outgoing nodes in projection
                (i.e there is a directed edge (i,j) when i is the key node and j is in the projection)
    :param dim: The dimension of the projection
    :return: Data for linear regression
    """
    nodes = []
    dict_node_params = {}
    for i in range(len(initial_proj_nodes)):
        node = initial_proj_nodes[i]
        if dict_enode_enode_in.get(node) is not None:
            x_1, z_11, z_12, k1, k2 = calculate_projection_of_neighbors(
                dict_enode_enode_in[node], dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim)
        else:
            x_1, z_11, z_12 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        if dict_enode_enode_out.get(node) is not None:
            x_2, z_21, z_22, k1, k2 = calculate_projection_of_neighbors(
                dict_enode_enode_out[node], dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim)
        else:
            x_2, z_21, z_22 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        a = np.zeros(dim*6)
        b = np.concatenate((x_1, x_2, z_11, z_12, z_21, z_22))
        if np.array_equal(a, b) is False:
            nodes.append(node)
            X = np.column_stack((x_1, x_2, z_11, z_12, z_21, z_22, np.ones(dim)))
            dict_node_params.update({node: X})
    return dict_node_params, nodes


def linear_regression(dict_params, nodes, dict_proj):
    """
    In order to find the best parameters to calculate a node's vector, we perform linear regression.
    :param dict_params: key==nodes in initial projection, value==A matrix of size 10*6
    :param nodes: The nodes that are in the initial projection
    :param dict_proj: Dict of projections (key==node, value==projection)
    :return: Best parameters to calculate the final projection.
    """
    all_x_list = []
    all_y_list = []
    for node in nodes:
        all_x_list.append(dict_params[node])
        all_y_list.append(dict_proj[node].T)
    X = np.concatenate(all_x_list)
    Y = np.concatenate(all_y_list)
    reg = LinearRegression()
    reg.fit(X, Y)
    params = reg.coef_
    reg_score = reg.score(X, Y)
    return params, reg_score


def main():
    # dimension of the embedding, can be changed
    dim = 10
    # if you want to see prints, do user_wish=True
    user_wish = False
    user_print("read the graph..", user_wish)

    # read the graph, here you can change to your own graph
    G = nx.read_edgelist("test.edgelist", create_using=nx.DiGraph(), delimiter=',')
    n = G.number_of_nodes()
    e = G.number_of_edges()
    print(n, e)

    # if you have a graph with labels, you can input its name here
    file = None

    # get the initial projection, number of nodes can be changed, see documentation of the function above
    initial_proj_nodes = get_initial_proj_nodes(G, 0.97)
    print(len(initial_proj_nodes))
    user_print("number of nodes in initial projection is: " + str(len(initial_proj_nodes)), user_wish)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    user_print("number of nodes in graph is: " + str(n), user_wish)
    user_print("number of edges in graph is: " + str(e), user_wish)
    # the nodes of our graph
    G_nodes = list(G.nodes())
    set_G_nodes = set(G_nodes)
    set_proj_nodes = set(initial_proj_nodes)
    G_edges = [list(i) for i in G.edges()]
    user_print("make a sub graph of the embedding nodes, it will take a while...", user_wish)
    # creating sub_G to do node2vec on it later
    sub_G = create_sub_G(initial_proj_nodes, G)
    user_print("calculate the projection of the sub graph with node2vec...", user_wish)
    # create dictionary of nodes and their projections after running node2vec on the sub graph
    dict_projections = our_node2vec(sub_G, dim)
    # from now count the time of our suggested algorithm
    t = time.time()
    neighbors_dict = create_dict_neighbors(G)
    set_nodes_no_proj = set_G_nodes - set_proj_nodes
    list_set_nodes_no_proj = list(set_nodes_no_proj)
    # create dicts of connections
    dict_node_node_in, dict_node_node_out, dict_node_enode, dict_enode_node, dict_enode_enode_in, dict_enode_enode_out\
        = create_dicts_of_connections(set_proj_nodes, set_nodes_no_proj, neighbors_dict, G)
    # calculate best parameters for the final projection calculation
    params_estimate_dict, labeled_nodes = crate_data_for_linear_regression(initial_proj_nodes, dict_projections, dict_enode_enode_in,
                                     dict_enode_enode_out, dim)
    pre_params, score = linear_regression(params_estimate_dict, labeled_nodes, dict_projections)
    print(pre_params, score)
    # create the final dictionary of nodes and their dictionaries
    final_dict_projections, set_no_proj = final_function(pre_params, dict_projections,
                                                    dict_node_enode, dict_enode_node, dict_node_node_out,
                                                    dict_node_node_in, dict_enode_enode_out,
                                                    dict_enode_enode_in, set_nodes_no_proj, 0.01, dim)
    # to calculate running time
    elapsed_time = time.time() - t
    print("running time: ", elapsed_time)
    print("The number of nodes that aren't in the final projection:", len(set_no_proj))
    print("The number of nodes that are in the final projection:", len(final_dict_projections))
    return final_dict_projections, G, file


final_dict_proj, G, file = main()