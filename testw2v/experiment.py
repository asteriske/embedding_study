import datetime
import gensim
import json
import os
import tensorflow as tf
from testw2v import config, google_example, gensim_utils, li2019, util
from typing import Any, Dict
conf = config.load()

class Experiment():

    def __init__(self, file: str, conf: Dict[str, Any]=None):
        pass

    def build_dataset(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run_all(self):
        
        start_time = datetime.datetime.now()

        self.build_dataset()
        self.build_model()
        self.fit()
        self.eval()

        self.runtime = (datetime.datetime.now() - start_time).seconds

        self.metrics['runtime_seconds'] = self.runtime
        self.metrics['conf'] = self.conf
        self.json = json.dumps(self.metrics)


class GensimExperiment(Experiment):

    def __init__(self, file:str, conf: Dict[str, Any]=None, no_op: bool=False):

        self.file = file
        self.vector_layer = None

        if conf is not None:
            self.conf = conf

        self.sequence_length = None
        if 'sequence_length' in self.conf.keys():
            self.sequence_length=self.conf['sequence_length']

        super(GensimExperiment).__init__()

        if not no_op:
            self.run_all()

    def build_dataset(self) -> None:



        self.sentences = gensim_utils.GensimSentenceGenerator(self.file,sequence_length=self.sequence_length)


    def build_model(self) -> None:     

        pass


    def fit(self):
        self.model = gensim.models.Word2Vec(sentences=self.sentences, 
                                            size=self.conf['size'],
                                            alpha=self.conf['alpha'],
                                            window=self.conf['window'],
                                            min_count=self.conf['min_count'],
                                            max_vocab_size=self.conf['max_vocab_size'],
                                            sample=self.conf['sample'],
                                            seed=self.conf['seed'],
                                            workers=self.conf['workers'],
                                            min_alpha=self.conf['min_alpha'],
                                            sg=self.conf['sg'],
                                            hs=self.conf['hs'],
                                            negative=self.conf['negative'],
                                            ns_exponent=self.conf['ns_exponent'],
                                            cbow_mean=self.conf['cbow_mean'],
                                            iter=self.conf['iter'],
                                            null_word=self.conf['null_word'],
                                            trim_rule=self.conf['trim_rule'],
                                            sorted_vocab=self.conf['sorted_vocab'],
                                            batch_words=self.conf['batch_words'],
                                            compute_loss=False,
                                            callbacks=(),
                                            max_final_vocab=self.conf['max_final_vocab']
                                            )

    def eval(self):
        self.metrics = gensim_utils.metrics(self.model.wv)


class GoogleExampleExperiment(Experiment):

    def __init__(self, file: str, conf: Dict[str, Any]=None, no_op: bool=False):

        self.file = file
        self.vector_layer = None

        if conf is not None:
            self.conf = conf

        super(GoogleExampleExperiment).__init__()

        if not no_op:
            self.run_all()


    def build_dataset(self) -> None:

        self.dataset, self.vector_layer = google_example.file_to_dataset(self.file, self.conf)

    
    def build_model(self):
         
         self.model = google_example.build_model()

    
    def fit(self):

        self.model.fit(self.dataset, epochs=self.conf['epochs'])


    def eval(self):
        vocab = self.vector_layer.get_vocabulary()
        weights = self.model.get_layer('w2v_embedding').get_weights()[0]

        gensim_obj = gensim_utils.w2v_to_gensim(vocab, weights)

        self.metrics = gensim_utils.metrics(gensim_obj)


class GoogleNewsExperiment(Experiment):

    def __init__(self, file: str, conf: Dict[str, Any]=None, no_op=False):

        if not os.path.exists('GoogleNews-vectors-negative300.bin.gz'):
            util.simple_download(conf['google_news_url'],'GoogleNews-vectors-negative300.bin.gz')

        self.file = file
        self.vector_layer = None

        if conf is not None:
            self.conf = conf

        super(GoogleNewsExperiment).__init__()

        if not no_op:
            self.run_all()


    def build_model(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)


    def eval(self):
        self.metrics = gensim_utils.metrics(self.model)

    
    def run_all(self):
        
        start_time = datetime.datetime.now()

        self.build_model()
        self.eval()

        self.runtime = (datetime.datetime.now() - start_time).seconds

        self.metrics['runtime_seconds'] = self.runtime
        self.metrics['conf'] = None
        self.json = json.dumps(self.metrics)


class Li2019Experiment(Experiment):

    def __init__(self, file: str, conf: Dict[str, Any]=None, no_op: bool=False):

        self.file = file
        self.conf = conf

        super(Li2019Experiment).__init__()

        if not no_op:
            self.run_all()


    def build_dataset(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        num_ns = self.conf['num_ns']
        vocab_size = self.conf['vocab_size']

        positive_matrix, negative_matrix, vocab = li2019.build_context_matrices(self.file, self.conf)

        self.dataset = li2019.build_pos_neg_generators(positive_matrix, negative_matrix, self.conf)


        self.vocab = vocab


    def build_model(self):

        self.model = li2019.Word2Vec(vocab_size=self.conf['vocab_size'], 
                                    embedding_dim=self.conf['embedding_dim'])


    def fit(self):

        li2019.generator_training_loop(self.model, self.dataset, self.conf)


    def eval(self):
        weights = self.model.get_layer('w2v_embedding').get_weights()[0]
        
        gensim_obj = gensim_utils.w2v_to_gensim(self.vocab, weights)

        self.metrics = gensim_utils.metrics(gensim_obj)