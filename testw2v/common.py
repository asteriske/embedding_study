import re
import string
import tensorflow as tf

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


def separate_labels(target, context, labels, conf):

    target.set_shape([conf['batch_size'],1])

    # context = tf.expand_dims(context,1)
    context.set_shape([conf['batch_size'],conf['num_ns']+1,1])

    labels.set_shape([conf['batch_size'],conf['num_ns']+1])
    return (
        (target, context), labels)