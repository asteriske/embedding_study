from collections import defaultdict
import logging
import numpy as np
import re
import string
from typing import Any, Dict, Generator, List, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

logger = logging.getLogger(__name__)
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


def row_to_contexts(row: np.array, context_dict: Dict[str, Any], window_size: int) -> None: 
    """
    Given a sequence of tokens update a context_dict containing their relationships. The 
    dict is updated by reference.
    """
    
    dense_row = [word for word in row if word != 0]
    
    row_len = len(dense_row)
    
    for idx, word in enumerate(dense_row):
        
        start_idx = idx - window_size
        if start_idx < 0:
            start_idx = 0
        end_idx = idx + 1 + window_size
        if end_idx > row_len:
            end_idx = row_len

        contexts = [dense_row[i] for i in range(start_idx, end_idx) if dense_row[i] != word]            

        for item in contexts:
            context_dict[word].append(item)                    

def build_context_matrices(file: str, conf: Dict[str, Any]) -> Tuple[np.array, np.array, List[str]]:

    logger.info("Beginning context matrix build...")
    pos_columns = conf['num_pos_columns']
    neg_columns = conf['num_neg_columns']
    sequence_length = conf['sequence_length']
    vocab_size = conf['vocab_size']
    window_size = conf['window_size']


    text_ds = (
        tf.data.TextLineDataset(file)
        .filter(lambda x: tf.cast(tf.strings.length(x), bool))
    )

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    vectorize_layer.adapt(text_ds)

    vector_ds = (
        # needs to be batched to work
        text_ds.batch(1024)
        .map(vectorize_layer)
        .unbatch()
    )

    #######################
    # Build positive matrix
    context_dict = defaultdict(list)

    for i in vector_ds:
        row_to_contexts(i.numpy(), context_dict, window_size=window_size)
    
    limited_context_dict = defaultdict(list)

    for k in context_dict.keys():
        limited_context_dict[k] = np.random.choice(context_dict[k], pos_columns)
    # 0 is the '' key but instead of replacing it with a word, maintain it to
    # keep the indexing nice and fill it with junk
    limited_context_dict[0] = np.random.choice(range(vocab_size), pos_columns)    

    context_matrix = np.stack([limited_context_dict[k] for k in range(vocab_size)])

    #######################
    # Build negative matrix

    difference_dict = defaultdict(list)

    for k in limited_context_dict.keys():
        unique_context_items = set(np.unique(limited_context_dict[k]))
        possible_negatives = set([x for x in range(vocab_size)])

        definite_negatives = possible_negatives.difference(unique_context_items)

        if len(definite_negatives) == 0:
            definite_negatives=[0]
        difference_dict[k] = np.random.choice(list(definite_negatives), neg_columns)
    
    negative_context_matrix = np.stack([difference_dict[k] for k in range(vocab_size)])

    logger.info("Context matrix build complete.")
    return context_matrix, negative_context_matrix, vectorize_layer.get_vocabulary()


def example_iterator_gen(positive_matrix: np.array, negative_matrix: np.array, conf) -> Generator[Tuple[tf.constant, tf.constant, tf.constant], None, None]:
    num_pos_columns = conf['num_pos_columns']
    num_neg_columns = conf['num_neg_columns']
    num_ns = conf['num_ns']
    vocab_size = conf['vocab_size']

    def generator_fn():    
        
        i = 0
        while i < num_pos_columns:
            if i % 1024 == 0:
                logger.info("Generator iteration: %s\r", i)
            
            tgt_ctx_label = (
                np.array(range(vocab_size)),
                positive_matrix[:,i],
                np.ones(vocab_size)
            )

            # We expect the negative matrix to be somewhat smaller than the 
            # positive matrix, so we'll loop over the negative several times
            # while iterating through the positive
            neg_start_idx = i*num_ns
            neg_end_idx = (i+1)*num_ns
            neg_indices = [x % num_neg_columns for x in range(neg_start_idx, neg_end_idx)]
            
            negative_ctx_label = (
                np.tile(np.array(range(vocab_size)),num_ns),
                negative_matrix[:,neg_indices].flatten('F'), # melt, maintain row order
                np.zeros(num_ns*(vocab_size))
            )
            
            example = (
                (tf.constant(np.concatenate([tgt_ctx_label[0], negative_ctx_label[0]]),shape=(vocab_size*(num_ns+1),1)),
                 tf.constant(np.concatenate([tgt_ctx_label[1], negative_ctx_label[1]]),shape=(vocab_size*(num_ns+1),1))),
                 tf.constant(np.concatenate([tgt_ctx_label[2], negative_ctx_label[2]]),shape=(vocab_size*(num_ns+1),1)),
            )
            yield example
            i+=1 

    return generator_fn


class Word2Vec(Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(vocab_size, 
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
    
        self.context_embedding = Embedding(vocab_size, 
                                           embedding_dim, 
                                           input_length=1,
                                           name='ctx_embedding')

        self.dots = Dot(axes=(2,2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)    
        dots = self.dots([ce, we])
        return self.flatten(dots)


def print_status_bar(iteration: int, total: int, loss: float, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)


def generator_training_loop(model: Word2Vec, dataset: tf.data.Dataset, conf: Dict[str,Any]) -> None:

    n_epochs = conf['epochs']
    num_pos_columns = conf['num_pos_columns']
    num_ns = conf['num_ns']

    dataset_iter = iter(dataset)
    num_steps = num_pos_columns * (num_ns + 1)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mean_loss = tf.keras.metrics.Mean()
    metrics = [tf.keras.metrics.BinaryAccuracy()]

    for epoch in range(1, n_epochs+1):
        logger.info("epoch{}/{}".format(epoch, n_epochs))

        for step in range(num_steps):
            (X_batch, y_batch) = dataset_iter.get_next()

            left_words, right_words = X_batch

            ## Train Left
            model.get_layer('ctx_embedding').trainable=False
            model.get_layer('w2v_embedding').trainable=True

            with tf.GradientTape() as tape:

                y_pred = model((left_words,right_words), training=True)

                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))

                loss = tf.add_n([main_loss], model.losses)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                mean_loss(loss)

                ## Train Right
                model.get_layer('ctx_embedding').trainable=True
                model.get_layer('w2v_embedding').trainable=False

                with tf.GradientTape() as tape:
                    y_pred = model((right_words, left_words), training=True)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))

                    loss = tf.add_n([main_loss], model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                mean_loss(loss)

                for metric in metrics:
                    metric(y_batch, y_pred)
            if step % 1024 == 0:
                print_status_bar(step, num_steps, mean_loss, metrics)
        print_status_bar(num_steps, num_steps, mean_loss, metrics)

        for metric in [mean_loss] + metrics:
            metric.reset_states()