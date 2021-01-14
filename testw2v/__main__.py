"""
Launch point for w2v runs. Construct a 'run()' object
which can build its own dataset, run it, and compute metrics.
"""

import os
from testw2v import config, experiment, util

conf = config.load()


def runner(exp_class, run_conf):


    if 'file' in run_conf.keys():
        file = run_conf['file']
    else:
        file = conf['file']

    experiment = exp_class(file, run_conf)

    return experiment.json 
    

if __name__ == "__main__":

    experiments = {
        'google_default': {
            'exp_class': experiment.GoogleExampleExperiment,
            'run_conf': conf['experiments']['google_default']},
         'gensim_default': {
            'exp_class': experiment.GensimExperiment,
            'run_conf': conf['experiments']['gensim_default']},
         'google_news': {
            'exp_class': experiment.GoogleNewsExperiment,
            'run_conf': conf['experiments']['google_news']},
         'gensim_max_final_vocab_4096': {
            'exp_class': experiment.GensimExperiment,
            'run_conf': conf['experiments']['gensim_max_final_vocab_4096']},
    }

    if not os.path.exists(conf['file']):
        util.prep_wikitext()

    runs = conf['runs']
    metrics = {}

    for run in runs:
        
        metrics[run] = runner(**experiments[run])

    util.write_metrics(path=conf['metrics_path'], metrics=metrics)