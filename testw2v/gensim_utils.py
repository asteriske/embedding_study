import gensim
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.test.utils import datapath
import numpy as np
import pandas as pd
from typing import Any, Dict, Generator, List

def w2v_to_gensim(vocab: List[str], weights: np.ndarray) -> Word2VecKeyedVectors:
    """
    Wrap word2vec weights in a gensim object for evaluation purposes
    """

    assert len(weights.shape) == 2, "Weights array is not 2D"

    size = weights.shape[1]

    gensim_vectors = Word2VecKeyedVectors(vector_size=size)

    gensim_vectors.add(entities=vocab, weights = weights.tolist())

    return gensim_vectors


def word_similarity(model: Word2VecKeyedVectors, word: str):
    """
    Find the closest words to a given word in a Gensim w2v model,
    ordered by cosine similarity
    """
    similarity_output = model.most_similar(positive=[word])
    df = pd.DataFrame(similarity_output).rename(columns={0:'word',1:'cosine_dist'})
    return df.to_dict('list')


def metrics(gensim_model: Word2VecKeyedVectors) -> Dict[str,Any]:

    metrics = {}
    metrics['french_similarity'] = word_similarity(gensim_model,'french')
    metrics['december_similarity'] = word_similarity(gensim_model,'december')
    metrics['street_similarity'] = word_similarity(gensim_model,'street')
    metrics['boat_similarity'] = word_similarity(gensim_model,'boat')

    wordsim353 = gensim_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'), case_insensitive=True)
    simlex999  = gensim_model.wv.evaluate_word_pairs(datapath('simlex999.txt'), case_insensitive=True)
    google     = gensim_model.wv.evaluate_word_analogies(datapath('questions-words.txt'), case_insensitive=True)

    benchmarks_df = pd.DataFrame({
        'test':['wordsim353','wordsim353','simlex999','simlex999','google'],
        'metric':['pearson_cor','spearman_cor','pearson_cor','spearman_cor','accuracy'],
        'value':[wordsim353[0][0], wordsim353[1][0], 
                 simlex999[0][0], simlex999[1][0],
                 google[0]]
    })
    
    metrics['benchmarks'] = benchmarks_df.to_dict('list')
    return metrics


class GensimSentenceGenerator():

    def __init__(self, file, sequence_length=None):
        self.file = file 
        self.sequence_length = sequence_length

    def __iter__(self) -> Generator[List[str], None, None]:
        for line in open(self.file):

            if not self.sequence_length:
                yield gensim.utils.simple_preprocess(line)
            else:
                yield gensim.utils.simple_preprocess(line)[0:self.sequence_length]

            