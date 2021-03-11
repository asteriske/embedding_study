import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten

from testw2v import common, gensim_utils
from testw2v.hash_embedding.hash_embedding import HashEmbedding
from testw2v.skipgram.skipgram import SkipgramV2, build_preprocess_vocab

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Word2Vec(tf.keras.Model):
  def __init__(self, embedding_width, num_words=1024, num_hash_buckets=2**20, num_hash_func=3):
    super(Word2Vec, self).__init__()

    self.target_embedding = HashEmbedding(num_hash_func=num_hash_func, 
            num_words=num_words, 
            num_hash_buckets=num_hash_buckets, # ~1MM 
            embedding_width=embedding_width, 
            random_seed=1139, 
            name="w2v_target_embedding") 
 
    self.context_embedding = HashEmbedding(num_hash_func=num_hash_func, 
            num_words=num_words, 
            num_hash_buckets=num_hash_buckets, # ~1MM 
            embedding_width=embedding_width, 
            random_seed=1139, 
            name="w2v_context_embedding") 
 
    self.dots = tf.keras.layers.Dot(axes=(3,2),dtype=tf.float32) 
    self.flatten = tf.keras.layers.Flatten() 
 
 
  def call(self, pair): 
    target, context = pair 
    we = self.target_embedding(target) 
    ce = self.context_embedding(context) 
    dots = self.dots([ce, we]) 
    return self.flatten(dots)

    
def build_dataset(file, conf):

    # I want to use the preprocessing function from a TextVectorization
    # layer like the one we're going to use later. The TextVectorization
    # object only assigns a function among many when it initializes the 
    # class, so I'll just make one, extract the function, and throw it out.
    _throwaway_vectorize_layer = TextVectorization(standardize=common.custom_standardization)
    preprocessing_fn = _throwaway_vectorize_layer._preprocess

    words_and_counts, total_words_sum, kept_words_sum = build_preprocess_vocab(file, preprocessing_fn, limit=conf['vocab_size'])

    # make room for '' and '[UNK]'
    word_counts = [0,0]+list(words_and_counts.values())[:-2]

    skipgram = SkipgramV2(window=conf['window_size'],
        vocab_size=conf['vocab_size'],
        frequencies=word_counts,
        num_negative_per_example=conf['num_ns'],
        sampling_threshold=conf['pos_sample_threshold'])

    vectorize_layer = TextVectorization(
        standardize=common.custom_standardization,
        max_tokens=conf['vocab_size'],
        output_mode='int',
        output_sequence_length=conf['sequence_length'],
        vocabulary=list(words_and_counts.keys())[:-2])
    
    text_vector_ds = (                
        tf.data.TextLineDataset(file)
        .filter(lambda x: tf.cast(tf.strings.length(x), bool))

        .batch(conf['batch_size'])
        .map(vectorize_layer)
        .cache()

        .map(skipgram)
        .unbatch()
        .batch(conf['batch_size'], drop_remainder=True)

        .map(lambda x,y,z: common.separate_labels(x,y,z,conf))
        .shuffle(500, reshuffle_each_iteration=True)
        .prefetch(AUTOTUNE)
    )

    return text_vector_ds, vectorize_layer.get_vocabulary()


def eval(model, vocab):

    num_words = len(vocab)

    realized_embedding_matrix = (
        # feed a single n*1 tensor in to look up all vocab words at once
        tf.squeeze(
            model.get_layer('w2v_target_embedding')(
                tf.expand_dims(tf.range(num_words),0)
            )
        ).numpy()
    )

    gensim_obj = gensim_utils.w2v_to_gensim(vocab, realized_embedding_matrix)

    return gensim_utils.metrics(gensim_obj)