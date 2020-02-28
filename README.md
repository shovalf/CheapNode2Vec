# Cheap Node2Vec
Calculating Node2Vec on few nodes only, then propagating to the whole graph. 

To see full description of the problem, its former solutions and our new suggested solutions, please read the "cheap node2vec survey.pdf"
file in the repository.

There are two suggested solutions: the first is for undirected graphs (cheap_node2vec.py) and the second is for directed graphs (directed_node2vec_final.py).
Both files are well documented and assume familiarity with the problem and suggested solutions.

In order to run both codes, one should get a graph in  '.edgelist' form and decide the number of nodes in the initial projection, and
change both of them in the main function. 
