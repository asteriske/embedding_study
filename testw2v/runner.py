import os
from testw2v import config, experiment, util
from testw2v.skipgram_google import skipgram_experiment

conf = config.load()

def runner(exp_class, run_conf):


    if 'file' in run_conf.keys():
        file = run_conf['file']
    else:
        file = conf['file']

    experiment = exp_class(file, run_conf)

    return experiment.json 
    

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
     'gensim_max_final_vocab_4096_len_10': {
        'exp_class': experiment.GensimExperiment,
        'run_conf': conf['experiments']['gensim_max_final_vocab_4096_len_10']},
     'google_mikolov_positives': {
         'exp_class': experiment.GoogleExampleExperiment,
         'run_conf': conf['experiments']['google_mikolov_positives']},
     'skipgram_google_default': {
         'exp_class': skipgram_experiment.SkipgramGoogleExperiment,
         'run_conf': conf['experiments']['skipgram_google_default']},
     'skipgramv2_google_default': {
         'exp_class': skipgram_experiment.SkipgramV2GoogleExperiment,
         'run_conf': conf['experiments']['skipgramv2_google_default']},
     'skipgramv2_google_small_batch': {
         'exp_class': skipgram_experiment.SkipgramV2GoogleExperiment,
         'run_conf': conf['experiments']['skipgramv2_google_small_batch']},
     'skipgramv2_google_small_batch_long_seq': {
         'exp_class': skipgram_experiment.SkipgramV2GoogleExperiment,
         'run_conf': conf['experiments']['skipgramv2_google_small_batch_long_seq']},
     'skipgramv2_google_small_batch_low_iter': {
         'exp_class': skipgram_experiment.SkipgramV2GoogleExperiment,
         'run_conf': conf['experiments']['skipgramv2_google_small_batch_low_iter']}
}
    
def main():


    if not os.path.exists(conf['file']):
        util.prep_wikitext()

    runs = conf['runs']
    metrics = {}

    for run in runs:
        
        metrics[run] = runner(**experiments[run])

    # util.write_metrics(path=conf['metrics_path'], metrics=metrics)