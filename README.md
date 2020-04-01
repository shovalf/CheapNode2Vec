# Cheap Node2Vec
Calculating Node2Vec on few nodes only, then propagating to the whole graph. 

To see full description of the problem, its former solutions and our new suggested solutions, please read the "cheap node2vec survey.pdf"
file in the repository.

There are two suggested solutions: the first is for undirected graphs (undirected_cheap_node2vec.py) and the second is for directed graphs (directed_cheap_node2vec.py).
Both files are well documented and assume familiarity with the problem and suggested solutions.

In order to run both codes, one should get a graph in  '.edgelist' or '.txt' form and decide the number of nodes in the initial projection, and change both of them in the main function. 

In order to check performance of the suggested method, we have some evaluation tasks:

1. Visualization task: 'visualization.py' file which commits a visualization task as explained in the survey file, and compare between regular node2vec and our method.
To do that, you need a graph in the format explained above, and can also supply nodes' labels if you have them.

2. Another Visualization Task: 'tensorbroad_visualization.py'

3. Node Classification task: 'node_classification.py' file which commits a node classification task as explained in the survey file, and compare performance of regular node2vec vs our method, and also compare between different number of nodes in the initial projection. In 'node_classification' file there is a full explanation of what is needed for the task and how to run it.

For example datasets, you can see labeled graphs in here: https://github.com/louzounlab/DataSets/tree/master/Graph/Node_Labeled_Grap.

Notice that the minimum number of nodes requested is 5K.
