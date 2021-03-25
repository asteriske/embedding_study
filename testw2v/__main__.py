"""
Launch point for w2v runs. Construct a 'run()' object
which can build its own dataset, run it, and compute metrics.
"""
import argparse
import os

from testw2v import config, moti, runner 

conf = config.load()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--build_moti_data',
        default=False, 
        required=False, 
        help='Path to store tranformed moti data',
        action='store')
    
    parser.add_argument('--moti_train_file',
        '-m',
        default=False, 
        required=False, 
        action='store_true')

    parser.add_argument('--evaluate_moti',
        default=False, 
        required=False, 
        action='store_true')

    args = parser.parse_args()

    if parser.build_moti_data:
        moti_exp = moti.MOTIExperiment(file=args.moti_train_file, 
            conf=conf['experiments']['moti'],
            no_op=True) 
        moti_exp.build_dataset()

        moti_exp.write_train_file(os.path.join(args.working_dir, 'moti_training.txt'))
        moti_exp.write_worddict(os.path.join(args.working_dir, 'vocab_file_raw'))

    if parser.evaluate_moti:
        

    runner.main()