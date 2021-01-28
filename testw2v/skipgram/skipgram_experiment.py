import tensorflow as tf
from testw2v import experiment, google_example
from testw2v.skipgram.skipgram import Skipgram
from typing import Any, Dict

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