import collections
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from typing import Tuple

def build_preprocess_vocab(file, preprocessing_fn, limit=None):
    """
    
    """        

    text_ds = tf.data.TextLineDataset(file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    
    count_dict = collections.defaultdict(int)

    text_ds_iter = iter(text_ds.map(preprocessing_fn))

    for i in text_ds_iter:

        values = K.get_value(i)
        for j in values.tolist():
            count_dict[j] += 1    

    sorted_count_dict = collections.OrderedDict({k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)})
    
    if not limit:
        limit = len(sorted_count_dict.items())
        
    kept_words_and_counts = collections.OrderedDict(list(sorted_count_dict.items())[0:limit])
    
    total_words_sum = sum([v for v in sorted_count_dict.values()])
    kept_words_sum = sum([v for v in kept_words_and_counts.values()])
    
    return kept_words_and_counts, total_words_sum, kept_words_sum




class Skipgram():
    """
    This function produces skipgrams for Word2Vec as defined in `https://www.tensorflow.org/tutorials/text/word2vec`.
    """

    def __init__(self, window: int,
                 vocab_size: int,
                 num_negative_per_example: int=4,
                 frequencies: tf.Tensor=None):

        self.frequencies = frequencies
        self.num_negative_per_example = num_negative_per_example
        self.vocab_size = vocab_size
        self.window = window


    def _downsample_by_table(self,
                             input: tf.Tensor, 
                             context_words: tf.Tensor) -> tf.Tensor:
        """
        Using an externally-provided probability vector, include
        words in the embedding with inverse frequency to expected
        occurrence.
        """
        input_shape = tf.shape(input) 
        print("input shape")
        print(input_shape)
        random_draws = tf.random.uniform(input_shape, dtype=tf.float64)
        print("random draws")
        print(random_draws)

        print('input') 
        print(input)
        print('self.frequencies')
        print(self.frequencies)
        matching_frequencies = tf.squeeze(tf.gather(self.frequencies, input))
        print('matching_frequencies')
        print(matching_frequencies)
        
        probability_less_than_draw = tf.squeeze(tf.cast(matching_frequencies > random_draws, tf.int64))
        print('probability_less_than_draw')
        print(probability_less_than_draw)
        
        # mask the missing words to 0
        downsampled_sequence = tf.math.multiply(input, probability_less_than_draw)
        print("downsampled_sequence")
        print(downsampled_sequence)
        
        # zero out the rows reflecting missing words
        # reshape to (n,1)
        downsampled_context = tf.reshape(probability_less_than_draw,[tf.size(probability_less_than_draw), 1]) * context_words
        print("downsampled_context")
        print(downsampled_context)
        
        return downsampled_sequence, downsampled_context


    def _make_fat_diagonal(self, size: int) -> tf.Tensor:
        """
        Produces a 2d (size,size) tensor in which all elements are
        zero, including the diagonal, excluding the offset diagonal
        of width (window-1) which is set to 1. Put another way, on the
        ith row the ith item is the target, and (window-1) items ahead
        are marked 1, as well as (window-1) items behind.
    
        _make_fat_diagonal(size=5, window=3)
    
        <tf.Tensor: shape=(5, 5), dtype=int32, numpy=
        array([[0, 1, 1, 0, 0],
               [1, 0, 1, 1, 0],
               [1, 1, 0, 1, 1],
               [0, 1, 1, 0, 1],
               [0, 0, 1, 1, 0]], dtype=int32)>
    
        """
        fat_ones = tf.linalg.band_part(
            tf.ones([size,size], dtype=tf.int64),
            num_lower=self.window,
            num_upper=self.window
        )
    
        return tf.linalg.set_diag(fat_ones, tf.zeros(size, dtype=tf.int64))


    def _make_positive_skipgrams(self, input: tf.Tensor) -> tf.Tensor:
        """
        tf_positive_skipgrams(sequence=tf.constant([1,2,3,4]), window=2)
    
        <tf.Tensor: shape=(6, 2), dtype=int32, numpy=
            array([[1, 2],
                   [2, 1],
                   [2, 3],
                   [3, 2],
                   [3, 4],
                   [4, 3]], dtype=int32)>
                   
        Each word is evaluated for frequency which may result in some target
        words being rejected.    
        """
    
        # Ensure the input is rank 2
        if tf.rank(input) == 1:
            input = tf.expand_dims(input, axis=0)
        input_shape = tf.shape(input)
        num_input_rows = input_shape[0]
        num_input_cols = input_shape[1]

        fat_diagonal = self._make_fat_diagonal(size=num_input_cols)

        expanded_input = tf.repeat(input, repeats=num_input_cols, axis=0)
        expanded_fat_diagonal = tf.tile(fat_diagonal, multiples=[num_input_rows, 1])

        # print("expanded_input")
        # print(expanded_input)
        context_words = tf.math.multiply(expanded_input, expanded_fat_diagonal)
        # print("context words")
        # print(context_words)
    
        # Apply table of probabilities. If a word in the sequence is too common,
        # it will be removed from the sequence and the corresponding row of 
        # context words will be removed as well. We wait to do it until now
        # because we want the common words to appear in the windows of other 
        # words.
    
        if self.frequencies is not None:
            downsampled_sequence, downsampled_context_words = self._downsample_by_table(input, context_words)
        else:
            downsampled_sequence = input
            downsampled_context_words = context_words
        # print("downsampled_sequence")
        # print(downsampled_sequence)
        # print("downsampled context words")
        # print(downsampled_context_words)
        # Unravel the sequence into a (n,1) tensor and repeat it so that
        # each sequence member is paired with an element of the context vector
        # to which it corresponds.
        key_and_context_with_zeros = tf.stack([
            tf.repeat(tf.reshape(downsampled_sequence,[-1]), num_input_cols),
            tf.reshape(downsampled_context_words, [-1]),
           ],axis=1)
        # print('key_and_context_with_zeros')
        # print(key_and_context_with_zeros)
        # we don't want rows where the target is 0 nor the context word is 0
        nonzero_rows = tf.where(tf.math.multiply(
            key_and_context_with_zeros[:,0],
            key_and_context_with_zeros[:,1]))
        # print('nonzero_rows')
        # print(nonzero_rows)

        key_and_context = tf.squeeze(tf.gather(key_and_context_with_zeros, nonzero_rows))
        # print("key and context")
        # print(key_and_context)

        if tf.rank(key_and_context) == tf.TensorShape([1]):
            return tf.expand_dims(key_and_context,0)
        
        return key_and_context


    def _select_skipgram_negatives(self, positive_samples: tf.Tensor) -> tf.Tensor:
        """
        Apply `draw_negative_samples` to every element of a (,1)-shaped tensor
        of positive examples.
        """
        
        def draw_negative_samples(x):
            
            matrix_x = tf.reshape(tf.cast(x, tf.int64), (1,1))
            neg_samples, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=matrix_x,
                                                                        num_true=1,
                                                                        num_sampled=self.num_negative_per_example,
                                                                        unique=True,
                                                                        range_max=self.vocab_size)
            return neg_samples            
    
        return tf.map_fn(fn=draw_negative_samples, elems=positive_samples)


    def _label_mask(self, negative_skipgrams: tf.Tensor) -> tf.Tensor:
        """
        Assume that data will take the form 
    
        [+, -, -, -]
        [+, -, -, -]
        [+, -, -, -]
    
        That is, a 2d tensor in which the first column is positive. 
    
        We can easily create a mask of labels to match from only the 
        dimensions of the negative example tensor:
    
        [1, 0, 0, 0]
        [1, 0, 0, 0]
        [1, 0, 0, 0]
        """
        num_rows = tf.shape(negative_skipgrams)[0]
        num_negative_cols = tf.shape(negative_skipgrams)[1]

        return(tf.concat([
            tf.ones((num_rows, 1),dtype=tf.int32),
            tf.zeros((num_rows, num_negative_cols),dtype=tf.int32)
        ], axis=1))


    def __call__(self, input: tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor]:

        positive_skipgrams = self._make_positive_skipgrams(input)
    
        negative_skipgrams = self._select_skipgram_negatives(positive_skipgrams[:, 1])
    
        labels = self._label_mask(negative_skipgrams)
    
        target = positive_skipgrams[:,0]
    
        features = tf.expand_dims(
            tf.concat([positive_skipgrams[:,1:2],
                       negative_skipgrams], axis=1),
                       axis=2)
        
        return target, features, labels
        # return positive_skipgrams, negative_skipgrams
        # return positive_skipgrams

class SkipgramV2():

    def __init__(self, window: int,
             vocab_size: int,
             frequencies: tf.Tensor,
             num_negative_per_example: int=4,
             sampling_threshold=0.001
             ):

        self.frequencies = tf.cast(frequencies,tf.int32)
        self.num_negative_per_example = num_negative_per_example
        self.sampling_threshold=sampling_threshold
        self.vocab_size = vocab_size
        self.window = window
        self._positive_probabilities = self._frequencies_to_probabilities()
        self._negative_distribution = self._compute_negative_probabilities(self.frequencies)

    # @staticmethod
    def _single_retention_probability(self, frac_of_corpus):
        return (
            min(
                1,
                (tf.sqrt(frac_of_corpus/self.sampling_threshold) + 1)*self.sampling_threshold/frac_of_corpus
            )
        )


    def _frequencies_to_probabilities(self):
        """
        Follow the example of Mikolov's code as outlined
        in http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        """

        # don't downsample without word frequencies
        if self.frequencies is not None:                
            return tf.ones(self.vocab_size, tf.float32)

        corpus_total = tf.reduce_sum(self.frequencies)

        frac_of_corpus = self.frequencies / corpus_total

        return tf.map_fn(self._single_retention_probability, frac_of_corpus)



    def _condense_sentence(self, words: tf.Tensor) -> tf.Tensor:
        """
        Condense full sentence sequences into blocks of selected words, arranged consecutively.
    
        Example:
    
        input = tf.constant([[4,0,0,2,3],
                             [5,2,3,0,2]])

        result = tf.constant([[4,2,3,0],
                              [5,2,3,2]])                         

        In cases where a probability vector is supplied:
    
        input = tf.constant([[4,0,0,2,3],
                             [5,2,3,0,2]])

        # chance of retaining word at index 5 is now 0
        retention_probabilities = tf.constant([0,1,1,1,1,0])  
    
        result = tf.constant([[4,2,3],
                              [2,3,2]])  
        """


        probs_per_word = tf.gather(self._positive_probabilities, words) # same size as words
    
    
        # Perform random draws against retention probabilities
        random_draws = tf.random.uniform(tf.shape(words),dtype=tf.float32)
        selected_words = probs_per_word > random_draws

        retention_indices = tf.where(selected_words)
        # subset words to only those selected -- produces 1d tensor
        selected_words = tf.gather_nd(words, retention_indices)
    
        # separate back into sentences by splitting 1d tensor into ragged tensor, then make unragged
        return (
            tf.RaggedTensor.from_value_rowids(selected_words, retention_indices[:,0])
            .to_tensor()
        )


    def _make_fat_diagonal(self, size: int) -> tf.Tensor:
        """
        Produces a 2d (size,size) tensor in which all elements are
        zero, including the diagonal, excluding the offset diagonal
        of width (window-1) which is set to 1. Put another way, on the
        ith row the ith item is the target, and (window-1) items ahead
        are marked 1, as well as (window-1) items behind.
    
        _make_fat_diagonal(size=5, window=3)
    
        <tf.Tensor: shape=(5, 5), dtype=int32, numpy=
        array([[0, 1, 1, 0, 0],
               [1, 0, 1, 1, 0],
               [1, 1, 0, 1, 1],
               [0, 1, 1, 0, 1],
               [0, 0, 1, 1, 0]], dtype=int32)>
    
        """
        
        fat_ones = tf.linalg.band_part(
            tf.ones([size,size], dtype=tf.int64),
            num_lower=self.window,
            num_upper=self.window
        )
    
        return tf.linalg.set_diag(fat_ones, tf.zeros(size, dtype=tf.int64))
        


    def _make_positive_skipgrams(self, input):

        # Ensure the input is rank 2
        if tf.rank(input) == 1:
            input = tf.expand_dims(input, axis=0)
        input_shape = tf.shape(input)
        num_input_rows = input_shape[0]
        num_input_cols = input_shape[1]

        sentence = self._condense_sentence(input)
        

        fat_diagonal = self._make_fat_diagonal(num_input_cols)


        # return fat_diagonal
        # print(f"fat diagonal\n{fat_diagonal}")
        expanded_input = tf.repeat(sentence, repeats=num_input_cols, axis=0)
        # print(f"expanded input\n{expanded_input}")
        repeated_input = tf.reshape(tf.repeat(sentence, repeats=num_input_cols, axis=1),[-1])
        expanded_fat_diagonal = tf.tile(fat_diagonal, multiples=[num_input_rows, 1])
        # print(f"expanded fat diagonal\n{expanded_fat_diagonal}")

        context_words = tf.math.multiply(expanded_input, expanded_fat_diagonal)
        # print(f"context words\n{context_words}")
        
        # Compose a column of targets and another column
        #of context words. For a 2d input tensor T of shape 
        # [m,n]:
        #
        # * create a target tensor by repeating every word 
        #   [i,j] n times, and reshape to a single column, 
        #   producing a 2d tensor of shape [m*n,1]
        #
        # * reshape the context word tensor (also of shape [m,n])
        #   so every element of the target tensor is paired with
        #   an element of the context tensor
        #
        # * remove all rows from the tensor where either item is 0
        # e.g. 
        #
        # original sentence: [1,2,3,4]
        #
        # fat diagonal, window 1:  [[0,1,0,0],
        #                           [1,0,1,0],
        #                           [0,1,0,1],
        #                           [0,0,1,0]]
        #
        # expanded_input:          [[1,2,3,4],
        #                           [1,2,3,4],
        #                           [1,2,3,4],
        #                           [1,2,3,4]]
        #
        # repeated_input: [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
        # 
        # context_words:           [[0,2,0,0],
        #                           [1,0,3,0],
        #                           [0,2,0,4],
        #                           [0,0,3,0]]
        #
        # targets (top 8/16):      [[1],
        #                           [1],
        #                           [1],
        #                           [1],
        #                           [2],
        #                           [2],
        #                           [2],
        #                           [2],
        #                            ...]
        # 
        # reshaped_context (top 8/16): [[0],
        #                               [2],
        #                               [0],
        #                               [0],
        #                               [1],
        #                               [0],
        #                               [3],
        #                               [0],
        #                               ....]
        #
        # target and context with zeros removed (top )
        #                        [[1,0], # remove
        #                         [1,2],
        #                         [1,0], # remove
        #                         [1,0], # remove
        #                         [2,1],
        #                         [2,0], # remove
        #                         [2,3],
        #                         [2,0], # remove
        #                         ....]

        key_and_context_with_zeros = tf.stack([
            repeated_input,
            tf.reshape(context_words, [-1]),
           ],axis=1)
        # print(f'key_and_context_with_zeros\n{key_and_context_with_zeros}')

        nonzero_rows = tf.cast(tf.math.multiply(
            key_and_context_with_zeros[:,0],
            key_and_context_with_zeros[:,1],)
        ,tf.bool)

        key_not_equal_context = key_and_context_with_zeros[:,0] != key_and_context_with_zeros[:,1]

        rows_to_keep = tf.where(
            tf.math.logical_and(
                nonzero_rows,
                key_not_equal_context
            )
        )

        key_and_context = tf.squeeze(tf.gather(key_and_context_with_zeros, rows_to_keep))
        # print(f"nonzero_rows\n{key_and_context}")

        # funny indexing to prevent loss of dimensions
        return key_and_context[:,0:1], key_and_context[:,1:2]


    @staticmethod
    def _compute_negative_probabilities(freqs, power=tf.constant(.75,dtype=tf.float32)):
        
        raised_to_power = tf.math.pow(tf.cast(freqs,dtype=tf.float32), power)
        dist = raised_to_power / tf.reduce_sum(raised_to_power)

        # Return a distribution object for easy sampling
        return tfp.distributions.Categorical(probs=dist,dtype=tf.int64)


    def _select_skipgram_negatives(self, target, context):
        
        num_rows = tf.shape(target)[0]
        # print(f"num negative rows: {num_rows}")

        negative_samples = self._negative_distribution.sample((num_rows, self.num_negative_per_example))

        # to reduce confusion, zero out elements of the negative
        # samples where they match either the target or the context
        # words
        target_spread = tf.tile(target, [1, self.num_negative_per_example])
        target_mask = tf.cast(negative_samples != target_spread, tf.int64)

        context_spread = tf.tile(context, [1, self.num_negative_per_example])
        context_mask = tf.cast(negative_samples != context_spread, tf.int64)

        # print(f"resulting mask: {tf.math.multiply(target_mask,context_mask)}")

        return tf.math.multiply(
            tf.math.multiply(negative_samples, target_mask), 
            context_mask)


    def _label_mask(self, negative_skipgrams: tf.Tensor) -> tf.Tensor:
        """
        Assume that data will take the form 
    
        [+, -, -, -]
        [+, -, -, -]
        [+, -, -, -]
    
        That is, a 2d tensor in which the first column is positive. 
    
        We can easily create a mask of labels to match from only the 
        dimensions of the negative example tensor:
    
        [1, 0, 0, 0]
        [1, 0, 0, 0]
        [1, 0, 0, 0]
        """
        num_rows = tf.shape(negative_skipgrams)[0]
        num_negative_cols = tf.shape(negative_skipgrams)[1]

        return(tf.concat([
            tf.ones((num_rows, 1),dtype=tf.int32),
            tf.zeros((num_rows, num_negative_cols),dtype=tf.int32)
        ], axis=1))


    def input_to_skipgram_format(self, input):

        target_words, context_words = self._make_positive_skipgrams(input)

        # return target_words, context_words

        # return self._make_positive_skipgrams(input)

        # print(f"target, context: {tf.stack([target_words, context_words],axis=1)}")
        negative_skipgrams = self._select_skipgram_negatives(target_words, context_words)
        # print(f"negative skipgrams {negative_skipgrams}")

        labels = self._label_mask(negative_skipgrams)

        features = tf.expand_dims(
            tf.concat([context_words,
            negative_skipgrams], axis=1),
            axis=2)

        return target_words, features, labels


    def __call__(self, input: tf.Tensor) -> tf.Tensor:

        return self.input_to_skipgram_format(input)