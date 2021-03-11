import tensorflow as tf
from tensorflow.lookup import StaticHashTable, KeyValueTensorInitializer


class HashEmbedding(tf.keras.layers.Layer):

    def _agg_add(self, tensors):
        print(tensors)
        return tf.math.add_n(tensors)
    
    def _agg_concat(self, tensors):
        print("concat tensors")
        print(tensors)
        return tf.concat(tensors,axis=2)
    
    def __init__(self, 
        num_hash_func=3, 
        num_words=10000, 
        num_hash_buckets=128, 
        embedding_width=128, 
        activation=None, 
        aggregation_mode='sum', 
        append_weight=False,
        random_seed=None,
        **kwargs):

        if random_seed is not None:
            tf.random.set_seed(random_seed)

        self.aggregation_mode=aggregation_mode
        self.append_weight=append_weight
        self.embedding_width=embedding_width
        self.num_hash_buckets=num_hash_buckets
        self.num_hash_func=num_hash_func
        self.num_words=num_words

        if self.aggregation_mode=='sum':
            self.aggregation_func = self._agg_add
        if self.aggregation_mode=='append':
            self.aggregation_func = self._agg_concat 

        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):

        self.hash_tables = []

        print("hash table values")
        for i in range(self.num_hash_func):
            values_tensor = tf.random.uniform(shape=[self.num_words],
                minval=0,
                maxval=(2**30)-1,
                dtype=tf.int32)
            # keys_tensor = tf.squeeze(tf.where(values_tensor))
            keys_tensor = tf.range(self.num_words, dtype=tf.int32)
            
            print((values_tensor % self.num_hash_buckets)+1)
            
            self.hash_tables.append(
                StaticHashTable(
                    initializer=KeyValueTensorInitializer(
                        keys=keys_tensor,
                        values=(values_tensor % self.num_hash_buckets)+1
                    ),
                    default_value=0
                )
            )
        print("===========================================================")
        print("===========================================================\n")
        
        # self.word_importance = tf.Variable(
        #     tf.random_normal_initializer(mean=0, stddev=0.0005)(shape=[self.num_words, self.num_hash_func], dtype=tf.float16)
        #     )
        self.word_importance = self.add_weight(shape=(self.num_words, self.num_hash_func),initializer=tf.random_normal_initializer(mean=0, stddev=0.0005), trainable=True, dtype=tf.float16)
        # self.word_importance = tf.Variable(
        #     tf.constant([[0,0,0,0],
        #                 [.1,.1,.1,.1],
        #                 [.2,.2,.2,.2],
        #                 [.3,.3,.3,.3],
        #                 [.4,.4,.4,.4]], dtype=tf.float16)
        # )
        print("word importance")
        print(self.word_importance)
        print("===========================================================")
        print("===========================================================\n")

        self.embedding_matrix = tf.Variable(
            tf.concat([
                tf.zeros(shape=[1, self.embedding_width], dtype=tf.float16),
                tf.random_normal_initializer(mean=0, stddev=0.1)(shape=[self.num_hash_buckets, self.embedding_width], dtype=tf.float16),
            ],0))

        # self.embedding_matrix = tf.Variable(
        #     tf.constant([[0,0,0],
        #                [1,1,1],
        #                [2,2,2],
        #                [3,3,3],
        #                [4,4,4],
        #                [5,5,5],
        #                [6,6,6],
        #                [7,7,7]],dtype=tf.float16)
        # )
        print('embedding matrix')
        print(self.embedding_matrix)
        print("===========================================================")
        print("===========================================================\n")

        super().build(batch_input_shape) # must be at end

    def call(self, X):

        word_ids_to_hash_space = tf.cast(X % self.num_words, tf.int32)
        print("Input % num_words")
        print(word_ids_to_hash_space)
        print("===========================================================")
        print("===========================================================\n")

        # We make sure the importance has a different id than the words. This way
        # if the words collide, the importances will not and we "lose" once, not twice.
        word_ids_to_hash_space_importance = (X+3) % self.num_words
        print("Input + 3 % num_words")
        print(word_ids_to_hash_space_importance)
        print("===========================================================")
        print("===========================================================\n")

        weighted_embeddings = []

        for hash_id, hash_tab in enumerate(self.hash_tables):
            # If a value in `word_ids_to_hash_space` is zero it won't find a value in 
            # the hash table since we've constrained the hash table to be > 0 and return
            # 0 only if there's a lookup at an index out of range for the hash table.
            
            # A value of 0 *is* out of range, so the hash will return zero if it gets
            # zero, and that will invoke the 0th entry of the embedding, which is by
            # construction full of zero.
            word_id_to_embedding_bucket = hash_tab.lookup(word_ids_to_hash_space)

            print("word_id_to_embedding_bucket")
            print(word_id_to_embedding_bucket)
            print("===========================================================")
            print("===========================================================\n")

            embedding_dims = tf.gather(self.embedding_matrix, word_id_to_embedding_bucket)
            print("looked up embedding")
            print(embedding_dims)
            print("===========================================================")
            print("===========================================================\n")

            importance = tf.gather(self.word_importance[:,hash_id], word_ids_to_hash_space_importance)
            print("found importance")
            print(importance)
            print("===========================================================")
            print("===========================================================\n")

            """
            The following operation is an element-wise multiplication between 
            tensors whose dimensions don't allow for matrix multiplication. We're
            multiplying the (i,j)th item of `importance` to the (i,j)th item in 
            `embedding_dims`, where the latter has a third `k` dimension which is
            the values of the looked-up embedding value. 
            
            For example:
            t3d = tf.constant([[[0,0],[1,1],[1,1]],
                               [[2,2],[3,3],[3,3]],
                               [[4,4],[5,5],[5,5]],
                               [[6,6],[7,7],[8,8]]])
            t3d
            <tf.Tensor: shape=(4, 3, 2), dtype=int32, numpy=
            array([[[0, 0],
                    [1, 1],
                    [1, 1]],

                   [[2, 2],
                    [3, 3],
                    [3, 3]],

                   [[4, 4],
                    [5, 5],
                    [5, 5]],

                   [[6, 6],
                    [7, 7],
                    [8, 8]]], dtype=int32)>
                
            t2d = tf.constant([[1,1,1],
                               [0,0,0],
                               [1,0,1],
                               [0,1,0]])
            t2d
            <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
            array([[1, 1, 1],
                   [0, 0, 0],
                   [1, 0, 1],
                   [0, 1, 0]], dtype=int32)>


            tf.einsum('ijk,ij->ijk',t3d,t2d)

            <tf.Tensor: shape=(4, 3, 2), dtype=int32, numpy=
            array([[[0, 0],
                    [1, 1],
                    [1, 1]],

                   [[0, 0],
                    [0, 0],
                    [0, 0]],

                   [[4, 4],
                    [0, 0],
                    [5, 5]],

                   [[0, 0],
                    [7, 7],
                    [0, 0]]], dtype=int32)>

            The "k" here might be more than one dimension, *e.g.*
            in the case where we look up multiple negative sampling
            dimensions at once. When this occurs, replace k with '...' 
            so
            
            'ijk,ij->ijk' 

            becomes

            'ij...,ij->ij...'
            
            """

            weighted_embeddings.append(

                    # tf.einsum('ij...,ij...->ij...',embedding_dims,importance))
                    tf.math.multiply(embedding_dims,tf.expand_dims(importance,axis=2)))



            print('adding to weighted_embeddings...')
            print(weighted_embeddings)
            print("===========================================================")
            print("===========================================================\n")

        weighted_average_embedding = self.aggregation_func(weighted_embeddings)
        print('weighted average embedding:')
        print(weighted_average_embedding)


        if self.append_weight:
            importances_this_word = tf.gather(self.word_importance, word_ids_to_hash_space_importance)
            print("weighted_emb")
            print(weighted_average_embedding)
            print("importances_this_word")
            print(importances_this_word)
            return tf.concat([weighted_average_embedding, importances_this_word],axis=2)

        return weighted_average_embedding 

    def compute_output_shape(self, input_shape):
        # return tf.TensorShape(input_shape[0], input_shape[1], self.embedding_width)

        weight_addition = 0

        if self.append_weight:
            weight_addition = self.num_hash_func

        if self.aggregation_mode == 'sum':
            return tf.TensorShape(input_shape[0], input_shape[1], self.embedding_width+weight_addition)
        else:
            return tf.TensorShape(input_shape[0], input_shape[1], (self.embedding_width*self.num_hash_functions)+weight_addition)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 
                "activation": tf.keras.activations.serialize(self.activation)}
