import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

class UniformIntInitializer(tf.keras.initializers.Initializer):

  def __init__(self, minval, maxval, modval):
    self.minval = minval
    self.maxval = maxval 
    self.modval = modval

  def __call__(self, shape, dtype=None, **kwargs):
    return tf.random.uniform(
        shape, minval=self.minval, maxval=self.maxval, dtype=dtype) % self.modval

  def get_config(self):  # To support serialization
    return {"minval":self.minval, "maxval":self.maxval, "modval":self.modval}


@tf.keras.utils.register_keras_serializable(package='Custom',name='hashembedding')
class HashEmbedding(tf.keras.layers.Layer):
    def _agg_add(self, weighted_embeddings):
        # -1 is along the embedding dim, but we want to 
        # sum the different embedding vectors for each input
        return tf.reduce_sum(weighted_embeddings, -2)
    

    def _agg_concat(self, weighted_embeddings):
        """
        For every input, concatenate the retreived weighted embedding
        tensors instead of summing them
        """
        return tf.concat(tf.unstack(weighted_embeddings, axis=-2),axis=2)


    def __init__(self, 
        num_hash_func=3, 
        num_words=10000, 
        num_hash_buckets=1000, 
        embedding_width=20, 
        activation=None, 
        aggregation_mode='sum', 
        append_weight=False,
        random_seed=None,
        input_length=None,
        **kwargs):    

        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        self.random_seed = random_seed
        if self.random_seed is not None:
            tf.random.set_seed(self.random_seed)

        self.aggregation_mode=aggregation_mode
        self.append_weight=append_weight
        self.embedding_width=embedding_width
        self.num_hash_buckets=num_hash_buckets
        self.num_hash_func=num_hash_func
        self.num_words=num_words
        self.input_length=input_length

        self.hash_table_init = UniformIntInitializer(minval=0,
            maxval=2**30,
            modval=self.num_hash_buckets)
        self.word_importance_init = tf.random_normal_initializer(mean=0,stddev=0.0005)

        self.embeddings_initializer = tf.random_normal_initializer(mean=0, stddev=0.1)

        if self.aggregation_mode=='sum':
            self.aggregation_func = self._agg_add
        if self.aggregation_mode=='append':
            self.aggregation_func = self._agg_concat 

        super(HashEmbedding, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)


    def build(self, input_shape=None):


        self.hash_table = self.add_weight(
            name='hash_table',
            shape=(self.num_words, self.num_hash_func),
            initializer=self.hash_table_init,
            dtype='int32',
            trainable=False
        )

        self.word_importance = self.add_weight(                
            name='word_importance',
            shape=(self.num_words, self.num_hash_func),
            initializer=self.word_importance_init,
            dtype='float32',
            trainable=True
        )

        self.embedding = self.add_weight(
            name='embedding',
            shape=(self.num_hash_buckets, self.embedding_width),
            initializer=self.embeddings_initializer,
            dtype='float32',
            trainable=True
        )

        self.built = True
    

    def call(self, X):

        word_ids_to_hash_space            = tf.cast(X % self.num_words, tf.int64)
        word_ids_to_hash_space_importance = tf.cast((X+3) % self.num_words, tf.int64)
    
        word_id_to_embedding_bucket = tf.nn.embedding_lookup(self.hash_table, word_ids_to_hash_space)
        importance_values           = tf.nn.embedding_lookup(self.word_importance, 
                                                             word_ids_to_hash_space_importance)
                                                             
        embedding_vectors   = tf.nn.embedding_lookup(self.embedding, word_id_to_embedding_bucket)
        weighted_embeddings = (tf.expand_dims(importance_values, -1) * embedding_vectors)

        aggregated_embeddings = self.aggregation_func(weighted_embeddings)
        
        if self.append_weight:
            return tf.concat([aggregated_embeddings, importance_values], axis=-1)
        else:
            return aggregated_embeddings           

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):

        weight_addition = 0

        if self.append_weight:
            weight_addition = self.num_hash_func

        if self.aggregation_mode == 'sum':
            return (input_shape[0], input_shape[1], self.embedding_width+weight_addition)
        else:
            return (input_shape[0], input_shape[1], (self.embedding_width*self.num_hash_func)+weight_addition)
        

    def get_config(self):
        config = {
          'num_hash_func': self.num_hash_func,
          'num_words': self.num_words,
          'num_hash_buckets': self.num_hash_buckets,
          'embedding_width': self.embedding_width,
          'activation': self.activation,
          'aggregation_mode': self.aggregation_mode,
          'append_weight': self.append_weight,
          'random_seed': self.random_seed,
          'input_length': self.input_length
        }
        base_config = super(HashEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))