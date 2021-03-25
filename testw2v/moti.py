from collections import deque, defaultdict
import mmh3
import numpy as np
import os
import time
from typing import Any, Dict, Tuple

import tensorflow as tf

from testw2v import common, experiment

AUTOTUNE = tf.data.experimental.AUTOTUNE


def pairify(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Transform a sequence tensor ['A','B','C','D','E']
    into target and context tensors along axis 1
    
    ([['C'],   [['A'],
      ['C'],    ['B'],
      ['C'],    ['D'],    
      ['C']],   ['E']]).
      
    Insensitive to input tensor length.
    """
    
    input_len = int(len(x))

    target_idx = int((input_len-1) / 2)
    
    targets = tf.reshape(tf.repeat(x[target_idx], repeats=(input_len-1),axis=0), (input_len-1,1))
    
    contexts = tf.reshape(tf.concat([x[0:target_idx],x[(target_idx+1):]],axis=0),
                          (input_len-1,1))
    
    
    return targets, contexts



def dataset_from_textfile():
    pass


def make_hash_function(seed, hash_space):
    """
    Returns a murmurhash3 hash function with the provided seed and hash space (range). Accepts
    singular or multiple inputs.

    :param seed: A seed value to distinguish different hash functions from each other
    :param hash_space: The size of the (positive) integer space we want to hash inputs into
    """
    def hash_function(item):

        def hash_one_element(x):
            return np.int64(mmh3.hash(x, seed=seed, signed=False) % hash_space)

        if isinstance(item, (list, tuple, np.ndarray)):
            return np.array([hash_one_element(y) for y in item], dtype=np.int64)
        else:
            return hash_one_element(item)

    return hash_function


# def read_vocab_file(vocab_file, vocab_size):
#     """
#     Reads the entries in the specified vocab file, and hashes them into the a range of the specified
#     vocab_size using h0. The input is a file with unhashed vocab entries, like this:

#     contentid,count
#     27:ejo2,41275
#     27:1hknq,7815
#     27:1hmkg,1653

#     To form the vocab_counts list we first hash each contentid to find the correct array position.
#     We then put the value 10000 in that array position in the list. After we finish reading the file
#     any unpopulated array positions get the value 1 assigned to them. Experimental results indicate
#     that using these values instead of the "real" counts works well. The RFC has more details.

#     :param vocab_file: A file containing the full vocabulary being used for training
#     :param vocab_size: The range to hash the vocabulary entries into
#     :return new_counts: A 1x[vocab_size] numpy array containing either a 1 or a 10000 in each slot
#     :return vocab_entries_count: The number of "active" vocabulary entries
#     """
#     logging.info("Updating vocab counts...")
#     vocab = pd.read_csv(vocab_file, index_col=None)

#     hash_word = make_hash_function(0, vocab_size)
#     hashed_contentids = hash_word(vocab['contentid'].values)

#     vocab_entries_count = hashed_contentids.size
#     logging.info(f"Read {vocab_entries_count} hashed content ids from {vocab_file}")

#     new_counts = np.array([1] * vocab_size)
#     new_counts[hashed_contentids] = 10000

#     return new_counts, vocab_entries_count




def make_hash_function(seed, hash_space):
    """
    Returns a murmurhash3 hash function with the provided seed and hash space (range). Accepts
    singular or multiple inputs.

    :param seed: A seed value to distinguish different hash functions from each other
    :param hash_space: The size of the (positive) integer space we want to hash inputs into
    """
    def hash_function(item):

        def hash_one_element(x):
            return np.int64(mmh3.hash(x, seed=seed, signed=False) % hash_space)

        if isinstance(item, (list, tuple, np.ndarray)):
            return np.array([hash_one_element(y) for y in item], dtype=np.int64)
        else:
            return hash_one_element(item)

    return hash_function


# def extract_batch_from_deques(target_deque, context_deque):
#     """
#     Extracts a batch (consisting of two parallel arrays of target and context words respectively)
#     from the provided deques by popping them off the end. Assumes that the deques contain at least
#     batch_size elements!

#     :param target_deque: The target words for each available training pair, hashed using H0
#     :param context_deque: The context words for each available training pair, hashed using H0
#     :return: two parallel arrays of batch_size elements, one for targets and one for contexts
#     """
#     target_list = []
#     context_list = []
#     for i in range(batch_size):
#         target_list.append(target_deque.pop())
#         context_list.append(context_deque.pop())

#     target_array = np.array(target_list, dtype=np.int64)
#     context_array = np.array(context_list, dtype=np.int64)

#     target_array.shape = batch_size
#     context_array.shape = (batch_size, 1)
#     return target_array, context_array


def get_negative_sampler(context_words, negative_samples, vocab_size, vocab_counts):
    """
    Returns sampler for negative examples using tensorflow's built-in
    fixed_unigram_candidate_sampler.

    Only sample hash values corresponding to urls. The vocab_file specifies which hash values to
    sample from. It is a csv where each line is [hash value, sample frequency] sorted by hash value
    from 0 to vocabulary_size - 1.

    The tf samplers do not let you sample from hash values with 0 probability, so as a first, naive
    implementation, we can set frequencies to 1 for hash values that do not map to a url, and 10000
    for hash values that do map to a url, so the sampler will heavily favor hash values with urls in
    the training data.

    The vocab_file should be updated when the training data changes, and * * IMPORTANT!!!! * * the
    update needs to propagate through the vocab_counts reference and into the graph. I really hope
    Tensorflow does not make its own copy of the vocab_counts list somewhere internally!
    """
    return tf.nn.fixed_unigram_candidate_sampler(
        true_classes=context_words,    # Positive training examples
        num_true=1,                    # The number of classes per training example
        num_sampled=negative_samples,  # Number of negative examples to sample
        unique=True,                   # Batches do not contain repeated examples
        range_max=vocab_size,          # Possible range of hash values to sample
        unigrams=vocab_counts)         # Sample frequencies for each hash value


def compute_embeddings(dataset, embedding_matrix, embedding_space, embedding_size, importance_params, size, scopename):
    """
    Generates the final embeddings for each member of the provided dataset, by applying the two hash
    functions H1 and H2, looking up the embeddings at those two positions, and then using the
    importance matrix to combine them into a final embedding. Details are in the RFC:

    https://docs.google.com/document/d/1e4Vbn7jFWg7wgJ8bx-utt4zgrQeZ_B6NnRzEpZxFs_Q/edit

    :param dataset: A data set (or batch thereof) from a tf.Dataset
    :param embedding_matrix: A tensor that stores an embedding matrix of size B x d, where
           B = the hash space size and d = the size of the embedding for each hash position
          (note than typically in our case B >> d)
    :param size: How many embeddings we want to calculate - either one batch, or all of them
    :param scopename: A string to help us keep track of what set of embeddings we are computing
    :return:
    """
    # We want to apply the 2 hash functions (which are referred to as H1 and H2 in the RFC, under
    # "Key Parts of the Hash Embedding Network") element-wise to every member of the input dataset.
    #
    # According to Patrick, these py_funcs are slow, and at some point we may want to think of
    # replacing them with something more efficient (and/or hashing upstream of here).
    #
    hash1 = tf.compat.v1.py_func(
        func=make_hash_function(1, embedding_space), inp=[dataset], Tout=tf.int64, name='hash1')

    hash2 = tf.compat.v1.py_func(
        func=make_hash_function(2, embedding_space), inp=[dataset], Tout=tf.int64, name='hash2')

    embedlookup1 = tf.nn.embedding_lookup(embedding_matrix, hash1, name='embed1')
    embedlookup2 = tf.nn.embedding_lookup(embedding_matrix, hash2, name='embed2')

    # Define the methodology to calculate the hash embedding for each member of the dataset.
    #
    with tf.name_scope(f'importance_param_{scopename}'):
        im_p0 = tf.nn.embedding_lookup(importance_params, dataset, name='im_p')
        im_p1 = tf.reshape(im_p0, [size, 2, 1, 1, 1])
        im_p2 = tf.transpose(im_p1, [1, 2, 0, 3, 4])
        im_p = tf.reshape(im_p2, [2, 1, 1 * size, 1], name='final_im_p')

    with tf.name_scope(f'stack_embed_{scopename}'):
        embed0 = tf.stack([embedlookup1, embedlookup2], axis=2, name='embed')
        embed1 = tf.reshape(embed0, [size, 2, embedding_size, 1])
        embed2 = tf.transpose(embed1, [1, 2, 0, 3])
        embed = tf.reshape(embed2, [1, 2, embedding_size, size * 1], name='final_embed')

    with tf.name_scope(f'conv_embed_{scopename}'):
        res0 = tf.nn.depthwise_conv2d(embed, im_p,
                                      strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        res1 = tf.reshape(res0, [1, embedding_size, size, 1, 1])
        res2 = tf.transpose(res1, [2, 0, 1, 3, 4])
        res = tf.reshape(res2, [size, embedding_size], name='final_conv2d')

    return res









class MOTIExperiment(experiment.Experiment):

    def __init__(self, file: str, 
        conf: Dict[str, Any]=None, 
        device=None, 
        no_op: bool=False,
        weights: np.array=None):

        self.file = file
        self.conf = conf        
        self.device = device

        self.dataset = None
        self.num_words = None
        self.vocab_counts = None

        if not no_op:
            self.run_all()


    def dataset_from_textfile(self) -> tf.data.Dataset:

        hash_word = make_hash_function(0, self.conf['vocab_size'])
        
        text_ds = tf.data.TextLineDataset(self.file)

        return (
            text_ds

            # Remove empty lines
            .filter(lambda x: tf.cast(tf.strings.length(x), bool))

            # Lower and strip punctuation
            .map(common.custom_standardization)

            # Make into ngrams, window_size = 3 means an n-gram 7 long including the target
            .map(tf.strings.split)
            .map(lambda x: tf.strings.ngrams(x, (self.conf['window_size']*2) - 1))
            .unbatch()

            # ngrams makes concated strings so we have split again
            .map(tf.strings.split)

            # divide into (batch, 1)-sized target and context tensors
            .map(pairify)
            .unbatch()
            .batch(self.conf['batch_size'])

            .prefetch(AUTOTUNE)
        )

    @staticmethod
    def vocab_counts_from_raw_file(filepath: str, vocab_size: int) -> Tuple[np.array, int]:
        """
        Achieve the goals of read_vocab_file (above) but generate
        from scratch on an unprocessed input file. 
    
         * Enumerate unique words
         * Hash each unique word
         * Count num distinct hashes
         * Produce a "counts" list where present hashes are counted as 10000 and nonpresent as 1
    
         There's probably a much faster way to do this, but this way I 
         achieve parity with the dataset feeding the ML.
        """
    
        text_ds = tf.data.TextLineDataset(filepath)
    
        ds = (
            text_ds
    
            # Remove empty lines
            .filter(lambda x: tf.cast(tf.strings.length(x), bool))
    
            # Lower and strip punctuation
            .map(common.custom_standardization)
    
            .map(tf.strings.split)
            .unbatch()
        )
    
        word_count_dict = defaultdict(list)
    
        for word in ds:
            try:
                word_count_dict[word.numpy()] += 1
            except TypeError:
                word_count_dict[word.numpy()] = 1
        
        hash_word = make_hash_function(0, vocab_size)
    
        hashed_ids = [hash_word(key)for key in list(word_count_dict.keys())]
    
        vocab_entries_count = len(hashed_ids)
        new_counts = np.array([1] * vocab_size)
        new_counts[hashed_ids] = 10000
    
        return new_counts, vocab_entries_count, word_count_dict


    def build_dataset(self):
        """
        Need to return both the tf dataset and also
        word counts to inform the unigram sampler in 
        sampled softmax.
        """

        self.dataset = self.dataset_from_textfile()
        self.num_words, self.vocab_counts, self.word_count_dict = self.vocab_counts_from_raw_file(self.file, self.conf['vocab_size'])


    def write_trainfile(self, outfile:str) -> None:

        with open(outfile,'w') as f:

            for batch in self.dataset:

                target, context = batch

                target_list = target.numpy().tolist()

                context_list = context.numpy().tolist()

                for i, x in enumerate(zip(target_list, context_list)):
                    f.write(x[0][0].decode()+'\t'+x[1][0].decode()+'\n')


    def write_worddict(self, outfile: str) -> None:

    
        with open(outfile,'w') as f:
            f.write('contentid,count\n')

            for k in self.word_count_dict.keys():
                if k.lower() not in ('nan','null'):
                    f.write(k.decode()+','+str(self.word_count_dict[k])+'\n')


    def create_tensorflow_graph(self) -> Tuple[tf.Graph, tf.compat.v1.train.Saver]:
        """
        Defines the Tensorflow graph that contains the model to be trained, which outputs the full
        embedding matrix. The variables in this graph have identifying names so that their prior state
        can be loaded from a checkpoint file in case the service is restarted.
    
        We specify that this graph should be trained on CPU, where we have observed that the service
        can handle approximately 7500 training pairs per second (in batches of 128). In the future, we
        may explore training on GPU, which might require some additional tuning.
    
        :return: the Graph itself, plus a Saver object that can be used to retrieve prior variable
                 states and to save down the current ones
        """
        batch_size = self.conf['batch_size']
        embedding_size = self.conf['embedding_dim']
        embedding_space = self.conf['embedding_space']
        negative_samples = self.conf['negative_samples']
        vocab_counts = self.vocab_counts
        vocab_size = self.conf['vocab_size']
        
    
        if self.device is None:
            device = '/cpu:0'
    
        graph = tf.Graph()
        with graph.as_default(), tf.device(device):
    
            # Define our training data generator: either a Kafka consumer (the default setting for
            # production), or a file reader if we have specified a file to read from.
            #
            # if training_file is None:
            #     generator_to_use = generate_training_data_from_kafka
            # else:
            #     generator_to_use = generate_training_data_from_file
    
            # Attach a Dataset to our generator, and define an iterator based on it. Note that the Saver
            # complains when it tries to serialize the iterator. It may be related to this issue:
            # https://github.com/tensorflow/tensorflow/issues/11679
            # I am leaving it alone for now, since the save + reload still works.
            #
            # dataset = (
            #     tf.data.Dataset
            #            .from_generator(generator_to_use,
            #                            output_types=(tf.int64, tf.int64),
            #                            output_shapes=(tf.TensorShape([batch_size]),
            #                                           tf.TensorShape([batch_size, 1]))))
    
            # my_iter = dataset.make_initializable_iterator()
            # my_iter = iter(self.dataset)
            # target_words, context_words = my_iter.get_next()
            target_words = tf.compat.v1.placeholder(tf.string, shape=[None,1])
            context_words = tf.compat.v1.placeholder(tf.string, shape=[None,1])
    
            # Declare our initializers and tensor shapes as variables, for brevity.
            #
            ru_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
            tn_init = tf.compat.v1.truncated_normal_initializer(stddev=1.0/np.sqrt(embedding_size))
            zr_init = tf.zeros_initializer
    
            emb_shape = [embedding_space, embedding_size]
            smw_shape = [vocab_size, embedding_size]
            smb_shape = [vocab_size]
            imp_shape = [vocab_size, 2]
    
            # Now declare our variables using the specified shapes and initializers.
            #
            embeddings = tf.compat.v1.get_variable("embeddings", shape=emb_shape, initializer=ru_init)
            softmax_weights = tf.compat.v1.get_variable("softmax_weights", shape=smw_shape, initializer=tn_init)
            softmax_biases = tf.compat.v1.get_variable("softmax_biases", shape=smb_shape, initializer=zr_init)
            importances = tf.compat.v1.get_variable("importance_params", shape=imp_shape, initializer=ru_init)
    
            # Get the embeddings for the current batch, then sample some negatives to go with them
            # and calculate the loss on the sample.
            #
            batch_embeddings = compute_embeddings(target_words, embeddings, embedding_space, embedding_size, importances, batch_size,
                                                  'train')
            negative_sampler = get_negative_sampler(context_words, negative_samples, vocab_size, self.vocab_counts)
    
            loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(
                        weights=softmax_weights,
                        biases=softmax_biases,
                        inputs=batch_embeddings,
                        labels=context_words,
                        num_sampled=negative_samples,
                        num_classes=vocab_size,
                        sampled_values=negative_sampler),
                    name='loss')
    
            # Define an optimizer to minimize the loss, and publish the loss to Tensorboard.
            #
            loss_summary = tf.summary.scalar("loss_summary", loss)
            optimizer = tf.train.AdagradOptimizer(1, name='optimizer').minimize(loss)
    
            # Now recompute the full embedding matrix, which is the final output of the graph, and
            # divide them by a normalizer to constrain the range.
            #
            domain = tf.constant(np.array(range(vocab_size)).astype(np.int64), name='domain')
            all_embeddings = compute_embeddings(domain, embeddings, importances, vocab_size, 'full')
    
            normalizer = tf.sqrt(tf.reduce_sum(tf.square(all_embeddings), axis=1, keepdims=True))
            normalized_embeddings = all_embeddings / normalizer
    
            # Register all of these objects as graph components.
            #
            for node in (optimizer, loss, normalized_embeddings, loss_summary, my_iter):
                graph.add_to_collection('nodes', node)
    
            # Also within the graph context, instantiate a Saver that will allow us to save and restore
            # the variables defined above.
            #
            saver = tf.compat.v1.train.Saver()
    
        return graph, saver
    
    
    def build_model(self):

        assert self.dataset is not None, "No dataset, perhaps needs to run build_dataset()"
        assert self.num_words is not None, "No num_words, perhaps needs to run build_dataset()"
        assert self.vocab_counts is not None, "No vocab_counts, perhaps needs to run build_dataset()"
        self.graph, self.saver = self.create_tensorflow_graph()

    
    def start_training_loop(self) -> None:
        """
        Starts the main training loop and trains continuously while the Kafka client listens for new
        training events. After processing each event, if enough time has passed, the training loop will
        save down a checkpoint of the current model state, and refresh the vocab frequency counts,
        before continuing with the next event.
        """

        # Load the vocab frequency counts, define the graph, and declare handles to access the nodes we
        # care about.
        #
        # update_vocab_counts()
        # graph, saver = create_tensorflow_graph()
        graph, saver = self.graph, self.saver
        optimizer, loss, normalized_embeddings, loss_summary, iterator = graph.get_collection('nodes')

        # Tag the run with the application startup time for Tensorboard's benefit. We can still
        # see all of the runs on one graph in Tensorboard, but it will use different colors for
        # different runs, so we will know when a restart occurred.
        #
        # startup_timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
        # writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, f'tb_{startup_timestamp}'))
        # writer.add_graph(graph)
        # writer.flush()

        step = 0
        interval_start = time.time()
        with tf.Session(graph=graph) as session:

            # If we have previously saved a checkpoint, restore our variables from it and resume
            # training where we left off. Otherwise, initialize the network from scratch.
            #
            # if len(os.listdir(checkpoint_dir)) > 0:
            #     info_log(f"Restoring from previous checkpoint in {checkpoint_dir}...")
            #     saver.restore(session, os.path.join(checkpoint_dir, "moti.ckpt"))
            #     info_log("Restore complete.")
            # else:
            #     info_log(f"No checkpoints found in {checkpoint_dir}, initializing the network.")
            #     tf.global_variables_initializer().run()

            # Initialize the dataset iterator that we will use for training, then start the loop. The
            # Kafka topic that supplies the Dataset's backing generator takes care of blocking the loop
            # while we wait for new data, so we can simply loop continuously here.
            #
            session.run(iterator.initializer)
            while True:
                try:
                    _, l, tensorboard_metrics = session.run([optimizer, loss, loss_summary])
                except tf.errors.OutOfRangeError:
                    # This means the generator has shut down, likely in response to a STOP control
                    # command. Exit the loop cleanly
                    # info_log("Training data generator has stopped; exiting training loop")
                    break

                step += 1
                # writer.add_summary(tensorboard_metrics, step)

                # If the checkpoint interval has passed, or if a CHECKPOINT control command has been
                # sent, save down a checkpoint of the current state of the network, reset the interval
                # start time, and log the time taken by the checkpoint operation to InfluxDB.
                #
                # current_seconds = time.time()
                # global checkpoint_flag
                # if checkpoint_flag or current_seconds - interval_start > checkpoint_interval_seconds:

                #     save_model_checkpoint(saver, session, normalized_embeddings)
                #     checkpoint_flag = False
                #     interval_start = current_seconds

                #     if influx_client is not None:
                #         duration_ms = 1000 * (time.time() - interval_start)
                #         influx_client.write_points([{"measurement": "checkpoint_time",
                #                                      "tags": {"host": fqdn},
                #                                      "fields": {"_time": duration_ms}}])

    def fit(self):

        self.start_training_loop()

    def eval(self):
        