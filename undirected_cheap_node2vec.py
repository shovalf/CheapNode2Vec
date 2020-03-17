import numpy as np
import networkx as nx
from node2vec import Node2Vec
import time
import heapq


def user_print(item, user_wish):
    """
    a function to show the user the state of the code. If you want a live update of the current state of the code and
    some details: set user wish to True else False
    """
    if user_wish is True:
        print(item, sep=' ', end='', flush=True)
        time.sleep(3)
        print(" ", end='\r')


def get_initial_proj_nodes(G, key):
    """
    function that gets the graph and return the nodes that we would like them to be
    in the initial projection
    """
    # a dictionary of the nodes and their degrees
    dict_degrees = dict(G.degree(G.nodes()))
    # a dictionary of the nodes and the average degrees
    dict_avg_neighbor_deg = nx.average_neighbor_degree(G)
    # sort the dictionary
    sort_degrees = sorted(dict_degrees.items(), key=lambda pw: (pw[1], pw[0]))  # list
    # sort the dictionary
    sort_avg_n_d = sorted(dict_avg_neighbor_deg.items(), key=lambda pw: (pw[1], pw[0]))  # list
    # choose only some percents of the nodes with the maximum degree
    top_deg = sort_degrees[int(key * len(sort_degrees)):len(sort_degrees)]
    # choose only some percents of the nodes with the maximum average degree
    top_avgn_deg = sort_avg_n_d[int(key * len(sort_avg_n_d)):len(sort_avg_n_d)]
    # a code to choose the nodes that have maximum degree and also maximum average degree
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
    creating a new graph from the final_nodes- so we can do the node2vec projection on it
    """
    sub_G = G.subgraph(list(proj_nodes))
    return sub_G


def our_node2vec(G, dim):
    """
    function to do node2vec embedding on graph G and return a dictionary of projections when
    value==node and key==projection
    """
    node2vec = Node2Vec(G, dimensions=dim, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    nodes = list(G.nodes())
    projections = {}
    for i in range(len(nodes)):
        projections.update({nodes[i]: np.asarray(model.wv.get_vector(nodes[i]))})
    return projections


def create_dict_neighbors(G):
    """
    create a dictionary when value==node and key==set_of_neighbors
    """
    G_nodes = list(G.nodes())
    neighbors_dict = {}
    for i in range(len(G_nodes)):
        node = G_nodes[i]
        neighbors_dict.update({node: set(G[node])})
    return neighbors_dict


def create_dicts_of_connections(set_proj_nodes, set_no_proj_nodes, neighbors_dict):
    """
    a function that creates 3 dictionaries:
    1. dict_node_node (explained below)
    2. dict_node_enode (explained below)
    2. dict_enode_enode (explained below)
    """
    # value == (node that isn't in the embedding), key == (set of its neighbours that are also not in the embedding)
    dict_node_node = {}
    # value == (node that isn't in the embedding), key == (set of neighbours thar are in the embedding)
    dict_node_enode = {}
    # key==(node that is in the projection and has neighbors in it), value==(set of neighbors that are in projection)
    dict_enode_enode = {}
    # nodes that are not in the projection
    list_no_proj = list(set_no_proj_nodes)
    list_proj = list(set_proj_nodes)
    for i in range(len(list_no_proj)):
        node = list_no_proj[i]
        # neighbors of the node that aren't in the projection
        set1 = neighbors_dict[node].intersection(set_no_proj_nodes)
        dict_node_node.update({node: set1})
        # neighbors of the node that are in the projection
        set2 = neighbors_dict[node].intersection(set_proj_nodes)
        if len(set2) > 0:
            dict_node_enode.update({node: set2})
    for i in range(len(list_proj)):
        node = list_proj[i]
        # neighbors of the node that are in the projection
        set1 = neighbors_dict[node].intersection(set_proj_nodes)
        if len(set1) > 0:
            dict_enode_enode.update({node: set1})
    return dict_node_node, dict_node_enode, dict_enode_enode


def calculate_projection(proj_nodes, dict_proj, dim, dict_enode_enode):
    proj = []
    mean_two_order_proj = []
    # get a list of the projections of the neighbors in the projection
    proj_nodes1 = list(proj_nodes)
    # the number of first order neighbors
    k1 = len(proj_nodes)
    k2 = 0
    # to calculate the mean projection of the second order neighbors
    for k in range(len(proj_nodes1)):
        two_order_neighs = dict_enode_enode.get(proj_nodes1[k])
        # if the neighbors in the projection also have neighbors in the projection calculate the average projection
        if two_order_neighs is not None:
            two_order_neighs = list(two_order_neighs)
            k2 += len(two_order_neighs)
            two_order_projs = []
            for i in range(len(two_order_neighs)):
                two_order_proj = dict_proj[two_order_neighs[i]]
                two_order_projs.append(two_order_proj)
            two_order_projs = np.array(two_order_projs)
            two_order_projs = np.mean(two_order_projs, axis=0)
        # else, the mean projection in 0
        else:
            two_order_projs = np.zeros(dim)
        mean_two_order_proj.append(two_order_projs)
        proj.append(dict_proj[proj_nodes1[k]])
    # for every neighbor we have the average projection of its neighbors, so now do average on all of them
    mean_two_order_proj = np.array(mean_two_order_proj)
    mean_two_order_proj = np.mean(mean_two_order_proj, axis=0)
    proj = np.array(proj)
    # find the mean proj
    proj = np.mean(proj, axis=0)
    # the final projection of the node
    final_proj = proj + 0.001 * (k2 / k1) * (proj - mean_two_order_proj)
    return final_proj


def one_iteration(dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e, current_node, dim):
    """
    a function that does one iteration over a given batch
    """
    condition = 1
    # get the neighbors in projection of node i
    embd_neigh = dict_node_enode[current_node]
    # the final projection of the node
    final_proj = calculate_projection(embd_neigh, dict_enode_proj, dim, dict_enode_enode)
    # add the node and its projection to the dict of projections
    dict_enode_proj.update({current_node: final_proj})
    # add our node to the dict of proj to proj and delete it from node_enode because now it's in the projection
    dict_enode_enode.update({current_node: embd_neigh})
    dict_node_enode.pop(current_node)
    # get the non embd neighbors of the node
    relevant_n_e = dict_node_node[current_node]
    # delete because now it is in the projection
    dict_node_node.pop(current_node)
    embd_neigh = list(embd_neigh)
    for i in range(len(embd_neigh)):
        f = dict_enode_enode.get(embd_neigh[i])
        if f is not None:
            dict_enode_enode[embd_neigh[i]].update([current_node])
        else:
            dict_enode_enode.update({embd_neigh[i]: set([current_node])})
    # check if num of non embd neighbors of our node bigger then zero
    if len(relevant_n_e) > 0:
        # loop of non embd neighbors
        relevant_n_e1 = list(relevant_n_e)
        for j in range(len(relevant_n_e)):
            tmp_append_n_n = dict_node_node.get(relevant_n_e1[j])
            if tmp_append_n_n is not None:
                # if relevant_n_e1[j] in dict_node_node:
                tmp_append_n_n = tmp_append_n_n-set([current_node])
                dict_node_node[relevant_n_e1[j]] = tmp_append_n_n
            tmp_append = dict_node_enode.get(relevant_n_e1[j])
            if tmp_append is not None:
                # add our node to the set cause now our node is in embd
                tmp_append.update(set([current_node]))
                dict_node_enode[relevant_n_e1[j]] = tmp_append
            else:
                dict_node_enode.update({relevant_n_e1[j]: set([current_node])})
    set_n_e.remove(current_node)
    return condition, dict_enode_proj, dict_node_enode, dict_node_node,dict_enode_enode, set_n_e


def final_function(dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e, batch_precent, dim):
    """
    the final function that iteratively divided the dictionary of nodes without embedding into number of batches
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
        # loop over node are not in the embedding
        if batch_size > len(set_n_e):
            num_times = len(set_n_e)
        else:
            num_times = batch_size
        list_n_e = list(set_n_e)
        heap = []
        for i in range(len(list_n_e)):
            my_node = list_n_e[i]
            a = dict_node_enode.get(my_node)
            if a is not None:
                num_neighbors = len(dict_node_enode[my_node])
            else:
                num_neighbors = 0
            heapq.heappush(heap, [-num_neighbors, my_node])
        for i in range(len(set_n_e))[:num_times]:
            # look on node number i in the loop
            current_node = heapq.heappop(heap)[1]
            if dict_node_enode.get(current_node) is not None:
                condition, dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e = one_iteration(dict_enode_proj,
                                                                                                   dict_node_enode,
                                                                                                   dict_node_node, dict_enode_enode,
                                                                                                   set_n_e,
                                                                                                   current_node, dim)
    return dict_enode_proj, set_n_e


def main():
    dim = 10  # dimension of the embedding, can be changed
    user_wish = True
    # if you want to see prints, do user_wish=True
    user_print("read the graph..", user_wish)
    G = nx.read_edgelist("test.edgelist", create_using=nx.DiGraph())
    # if you have a graph with labels, you can input its name here
    file = None
    # get the initial projection by set and list to help us later
    initial_proj_nodes = get_initial_proj_nodes(G, 0.95)
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
    dict_projections = our_node2vec(sub_G, dim)
    # convert the graph to undirected
    H = G.to_undirected()
    # from now count the time of our suggested algorithm
    t = time.time()
    neighbors_dict = create_dict_neighbors(H)
    # making all lists to set (to help us later in the code)
    set_nodes_no_proj = set_G_nodes - set_proj_nodes
    list_set_nodes_no_proj = list(set_nodes_no_proj)
    # create dicts of connections
    dict_node_node, dict_node_enode, dict_enode_enode = create_dicts_of_connections(set_proj_nodes, set_nodes_no_proj,
                                                                                    neighbors_dict)
    # the final function to get the embeddings
    final_dict_enode_proj, set_n_e = final_function(dict_projections, dict_node_enode, dict_node_node, dict_enode_enode, set_nodes_no_proj, 0.01, dim)
    # to calculate running time
    elapsed_time = time.time() - t
    print("running time: ", elapsed_time)
    print("The number of nodes that aren't in the final projection:", len(set_n_e))
    print("The number of nodes that are in the final projection:", len(final_dict_enode_proj))
    return final_dict_enode_proj, G, file


final_dict_proj, G, file = main()