import collections
import itertools
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten

from testw2v import common, gensim_utils
from testw2v.hash_embedding.hash_embedding import HashEmbedding
from testw2v.skipgram.skipgram import SkipgramV2, build_preprocess_vocab

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from typing import Dict, Tuple

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Word2Vec(tf.keras.Model):
    
    def __init__(self, 
        vocab: Dict[int,int],
        batch_size: int=64,
        embedding_width: int=20, 
        num_words: int=int(1e6), 
        num_hash_buckets: int=int(1e5), 
        num_hash_func: int=2, 
        num_negative: int=4,                    
        ):

        super(Word2Vec, self).__init__()

        self.batch_size            = batch_size
        self.num_hash_buckets      = num_hash_buckets
        self.num_negative          = num_negative
        self.num_words             = num_words
        self.vocab = vocab
        self.negative_distribution = self._vocab_to_negative_dist()


        self.target_embedding = HashEmbedding(
            num_hash_func=num_hash_func, 
            num_words=num_words, 
            num_hash_buckets=num_hash_buckets, 
            embedding_width=embedding_width, 
            random_seed=1139, 
            name="w2v_target_embedding") 

        self.context_embedding = HashEmbedding(
            num_hash_func=num_hash_func, 
            num_words=num_words,               # K ~ 2.5MM 
            num_hash_buckets=num_hash_buckets, # B ~ 250k
            embedding_width=embedding_width,   # d ~ 20 
            random_seed=1139, 
            name="w2v_context_embedding") 

        self.dots = tf.keras.layers.Dot(axes=(2,2), dtype=tf.float32) 
        self.flatten = tf.keras.layers.Flatten() 


    def call(self, pair): 
        
        target, context = pair
        
        we = self.target_embedding(
            self._hash_target(target) )                
        
        ce = self.context_embedding(
            self._prep_context(context) )
        
        dots = self.dots([ce, we]) 
        
        return self.flatten(dots)


    def _vocab_to_negative_dist(self, power=tf.constant(.75, dtype=tf.float32)):
        """
        """

        freqs = [self.vocab[x] for x in range(self.num_words)]

        raised_to_power = tf.math.pow(tf.cast(freqs, dtype=tf.float32), power)

        dist = raised_to_power + 1.0#/ tf.reduce_sum(raised_to_power)

        return tf.cast(dist, tf.int64).numpy().tolist()

    
    def _hash_target(self, x):
        # function D_1
        hashed_values = tf.strings.to_hash_bucket_fast(x, self.num_words)
        
        return hashed_values


    def _prep_context(self, x):
        """
        TODO: see if masking slows it down much
        """

        context_idx = tf.strings.to_hash_bucket_fast(x, self.num_words)
        
        negative_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes = context_idx,
            num_true = 1,
            num_sampled = (self.num_negative * self.batch_size),
            unique = False,
            range_max = self.num_words,
            unigrams = self.negative_distribution
        )
        
        return tf.concat([
            context_idx, 
            tf.reshape(negative_samples,(tf.shape(x)[0], self.num_negative))
            ], axis=1)
            
    
def group_and_label(target, context, num_ns=4):
    
    return ((target, context), tf.constant([1]+[0]*num_ns, dtype=tf.int32))


def skipgram_window(target_idx, seq, window):
    
    return [(seq[target_idx], seq[j]) 
            for j in range(max(0,target_idx-window), min(len(seq), target_idx+window)) if j!=target_idx]
    

def skipgram(seq, window=3):
    """
    Stick together lists of lists into a single list
    """
    
    return list(itertools.chain(*[skipgram_window(i, seq, window) for i in range(len(seq))]))                        


def line_generator_maker(textfile, window):
    def line_generator():

        with open(textfile, 'r') as f:
            for line in f.readlines():
                sequence = tf.keras.preprocessing.text.text_to_word_sequence(line)
                for elem in skipgram(sequence, window):
                    yield elem
                                
    return line_generator


def build_vocab(file: str) -> Dict[str, int]:
    
    dd = collections.defaultdict(int)
    
    with open(file,'r') as f:
        for line in f.readlines():
            sequence = tf.keras.preprocessing.text.text_to_word_sequence(line)
            
            for word in sequence:
                dd[str(word)] += 1
                
    return dd
 

def text_vocab_to_hash(text_vocab, num_words) -> Dict[int,int]:

    hash_vocab = collections.defaultdict(int)

    for k in text_vocab.keys():
        hash_vocab[tf.strings.to_hash_bucket_fast(k, num_words).numpy()] += text_vocab[k] 

    return hash_vocab


def _parse_function(example_proto):

    feature_desc = {
        'target': tf.io.FixedLenFeature([],tf.string, default_value=''),
        'context': tf.io.FixedLenFeature([],tf.string, default_value=''),
    }

    return (tf.io.parse_single_example(example_proto, feature_desc))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(target, context):
    feature = {
        'target': _bytes_feature(target.encode()),
        'context': _bytes_feature(context.encode())
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(line_gen):
    my_lg = line_gen()
    with tf.io.TFRecordWriter('my.tfrecord') as writer:
        for i, elem in enumerate(my_lg):

            writer.write(serialize_example(elem[0],elem[1]))

            if i % 500000 == 0:
                print(f"{i/int(1e6)}MM")


def build_dataset(file, conf) -> Tuple[tf.data.Dataset, Dict[int,int]]:

    line_gen = line_generator_maker(textfile=file, window=conf['window_size'])

    if conf['write_file']:
        write_tfrecord(line_gen)

    text_vocab = build_vocab(file)

    hash_vocab = text_vocab_to_hash(text_vocab, conf['he_importance_vector_params'])

    tf_data = (
        tf.data.TFRecordDataset('my.tfrecord')
        .map(_parse_function)
        .map(lambda x: group_and_label(x['target'],x['context'], conf['num_ns']))
        .shuffle(int(1e4), reshuffle_each_iteration=True)
        .batch(conf['batch_size'], drop_remainder=True)
        .prefetch(AUTOTUNE)
    )


    return tf_data, hash_vocab


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