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


# test with random mbeddings :
if __name__ == '__main__':
    embedding_vectors = np.random.rand(1000,100)
    log_tensorboard(embedding_vectors,'RandomProjection','./tensorboard/')