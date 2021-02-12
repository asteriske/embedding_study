import re
import string
from numpy import vectorize
import tensorflow as tf
from testw2v import experiment, google_example
from testw2v.skipgram.skipgram import Skipgram, SkipgramV2, build_preprocess_vocab
from typing import Any, Dict
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


AUTOTUNE = tf.data.experimental.AUTOTUNE

class SkipgramGoogleExperiment(experiment.GoogleExampleExperiment):


    def build_dataset(self) -> None:
        # self.file, self.conf
        self.skipgram_layer = Skipgram(window=self.conf['window_size'], 
            vocab_size=self.conf['vocab_size'],
            num_negative_per_example=self.conf['num_ns'])
        
        text_ds = tf.data.TextLineDataset(self.file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        vectorize_layer = google_example.prepare_vectorize_layer(text_ds)

        def separate_labels(target, context, labels):

            target.set_shape([self.conf['batch_size']])

            # context = tf.expand_dims(context,1)
            context.set_shape([self.conf['batch_size'],self.conf['num_ns']+1,1])

            labels.set_shape([self.conf['batch_size'],self.conf['num_ns']+1])
            return (
                (target, context), labels)

        text_vector_ds = (
            text_ds
            .batch(self.conf['batch_size'])
            .prefetch(AUTOTUNE)
            .map(vectorize_layer)
            .map(self.skipgram_layer)
            .unbatch()
            .batch(self.conf['batch_size'], drop_remainder=True)
            .map(separate_labels)        
            .cache()
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        self.dataset = text_vector_ds
        self.vector_layer = vectorize_layer

class SkipgramV2GoogleExperiment(experiment.GoogleExampleExperiment):

    @staticmethod
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                                      '[%s]' % re.escape(string.punctuation), '')

    def separate_labels(self, target, context, labels):

        target.set_shape([self.conf['batch_size'],1])

        # context = tf.expand_dims(context,1)
        context.set_shape([self.conf['batch_size'],self.conf['num_ns']+1,1])

        labels.set_shape([self.conf['batch_size'],self.conf['num_ns']+1])
        return (
            (target, context), labels)


    def build_dataset(self) -> None:

        # I want to use the preprocessing function from a TextVectorization
        # layer like the one we're going to use later. The TextVectorization
        # object only assigns a function among many when it initializes the 
        # class, so I'll just make one, extract the function, and throw it out.
        _throwaway_vectorize_layer = TextVectorization(standardize=self.custom_standardization)
        preprocessing_fn = _throwaway_vectorize_layer._preprocess

        # preprocessing_fn = self.custom_standardization

        words_and_counts, total_words_sum, kept_words_sum = build_preprocess_vocab(self.file, preprocessing_fn, limit=self.conf['vocab_size'])

        # make room for '' and '[UNK]'
        word_counts = [0,0]+list(words_and_counts.values())[:-2]

        self.skipgram = SkipgramV2(window=self.conf['window_size'],
            vocab_size=self.conf['vocab_size'],
            frequencies=word_counts,
            num_negative_per_example=self.conf['num_ns'])

        vectorize_layer = TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=self.conf['vocab_size'],
            output_mode='int',
            output_sequence_length=self.conf['num_ns'],
            vocabulary=list(words_and_counts.keys())[:-2])
        
        text_vector_ds = (                
            tf.data.TextLineDataset(self.file)
            .filter(lambda x: tf.cast(tf.strings.length(x), bool))

            .batch(self.conf['batch_size'])
            .map(vectorize_layer)
            .cache()

            .map(self.skipgram)
            .unbatch()
            .batch(self.conf['batch_size'], drop_remainder=True)

            .map(self.separate_labels)
            .shuffle(500, reshuffle_each_iteration=True)
            .prefetch(AUTOTUNE)
        )

        self.dataset = text_vector_ds
        self.vector_layer = vectorize_layer