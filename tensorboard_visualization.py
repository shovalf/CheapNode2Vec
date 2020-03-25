import os
import tensorflow as tf
import numpy as np
from tensorboard.plugins import projector

metadata_file = 'metadata.tsv'

def log_tensorboard(embedding_vectors, embedding_name,log_dir, metadata=None):

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    tensor_embedding = tf.Variable(embedding_vectors, name=embedding_name)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_embedding.name

    # set metadata
    if metadata is not None:
        with open(os.path.join(log_dir,metadata_file), 'w') as f:
            for label in metadata:
                f.write(f'{label}\n')

        embedding.metadata_path = metadata_file

    # Specify sprite for image embeddings
    # embedding.sprite.image_path = path_for sprites image
    # embedding.sprite.single_image_dim.extend([28, 28]) sprite thumbnail size

    projector.visualize_embeddings(log_dir, config)
    saver = tf.compat.v1.train.Saver([tensor_embedding])  # Must pass list or dict
    saver.save(sess=None, global_step=0, save_path=log_dir)


'''
using example:
1. load graph 
2. run cheap node2vec 
3. save nodes vectors (and optionaly labels) to tensorboard logs 
'''

from directed_cheap_node2vec import main

log_dir = r'./tensorboard/'


if __name__ == '__main__':

    list_dicts, G, file = main()
    dict_proj = list_dicts[0]

    # load labels file
    with open(file) as f:
        labels_list = [label.split() for label in f]

    # create labels list in nodes order :
    labels_dict = {label[0]:label[1] for label in labels_list}
    labels = [labels_dict[node] for node in dict_proj.keys()]

    projections_cheap = [dict_proj[key] for key in dict_proj.keys()]
    log_tensorboard(projections_cheap, 'projections_cheap', log_dir=log_dir, metadata=labels)

    print('for visualization run:')
    print(f'tensorboard --logdir {log_dir}')

