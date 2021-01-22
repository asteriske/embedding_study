import math
import re
import string
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tqdm

from testw2v import config
conf = config.load()['experiments']['google_default']

conf_node = conf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def count_vocab(text_ds, vocab_size):
    
    total_words = 0
    
    vocab_counts = [0]*vocab_size

    for i, x in enumerate(text_ds):

        for w in x.numpy():
            if w != 0:
                vocab_counts[w] += 1
                total_words += 1

    return vocab_counts, total_words


def subsample(word_count, total_counts):
    """
    The positive subsampling strategy of Mikolov et al, 2013
    """

    frac_this_word = word_count / total_counts
    
    if word_count == 0:
        return 0
    
    p_w = ((math.sqrt(frac_this_word/0.001)) + 1)*(0.001/frac_this_word)
    return min(p_w,1)    


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed, retention_probs):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  if not retention_probs:
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
  else:
    sampling_table = retention_probs
  
  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence, 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=seed, 
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


def prepare_vectorize_layer(text_ds):

    vectorize_layer = TextVectorization(
        # standardize='lower_and_strip_punctuation',
        standardize=custom_standardization,
        max_tokens=conf['vocab_size'],
        output_mode='int',
        output_sequence_length=conf['sequence_length'])

    vectorize_layer.adapt(text_ds.batch(conf['batch_size']))

    return vectorize_layer


def vectorize_text(path_to_file, zipped=False):

    text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    vectorize_layer = prepare_vectorize_layer(text_ds)

    text_vector_ds = (
        text_ds
        .batch(conf['batch_size'])
        .prefetch(AUTOTUNE)
        .map(vectorize_layer)
        .unbatch()
    )

    if zipped:
        return tf.data.Dataset.zip((text_ds, text_vector_ds))

    retention_probs = None

    if 'mikolov_positive_sample' in conf.keys():
        if conf['mikolov_positive_sample'] == True:
            vocab_counts, total_words = count_vocab(text_vector_ds, conf['vocab_size'])
            retention_probs = [subsample(x, total_words) for x in vocab_counts]
    
    return text_vector_ds, vectorize_layer, retention_probs


def sequences_to_dataset(text_dataset, retention_probs=None):

    sequences = list(text_dataset.as_numpy_iterator())
    targets, contexts, labels = generate_training_data(
        sequences=sequences, 
        window_size=conf['window_size'], 
        num_ns=conf['num_ns'], 
        vocab_size=conf['vocab_size'], 
        seed=conf['seed'],
        retention_probs=retention_probs)

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(conf['buffer_size']).batch(conf['batch_size'], drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset


def file_to_dataset(file_path):
    sequences, vectorize_layer, retention_probs = vectorize_text(file_path)
    dataset = sequences_to_dataset(sequences, retention_probs=retention_probs)

    return dataset, vectorize_layer


class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=conf['num_ns']+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)


# def custom_loss(x_logit, y_true):
#       return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


def build_model():
    embedding_dim = conf['embedding_dim'] 
    word2vec = Word2Vec(conf['vocab_size'], embedding_dim)
    word2vec.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return word2vec
