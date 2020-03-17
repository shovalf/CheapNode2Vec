# Cheap Node2Vec
Calculating Node2Vec on few nodes only, then propagating to the whole graph. 

To see full description of the problem, its former solutions and our new suggested solutions, please read the "cheap node2vec survey.pdf"
file in the repository.

There are two suggested solutions: the first is for undirected graphs (undirected_cheap_node2vec.py) and the second is for directed graphs (directed_cheap_node2vec.py).
Both files are well documented and assume familiarity with the problem and suggested solutions.

In order to run both codes, one should get a graph in  '.edgelist' or 'txt' form and decide the number of nodes in the initial projection, and change both of them in the main function. 

In order to check performance of the suggested method, one can run the 'GEM_visualization' file which commits a visualization task as explained in the survey file, and compare between regular node2vec and our method.
To do that, you need a graph in the format explained above, and can also supply nodes' labels if you have them.

For example datasets, you can see labeled graphs in here: https://github.com/louzounlab/DataSets/tree/master/Graph/Node_Labeled_Grap.

Notice that the minimum number of nodes requested is 10K.
