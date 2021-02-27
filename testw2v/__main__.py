"""
Launch point for w2v runs. Construct a 'run()' object
which can build its own dataset, run it, and compute metrics.
"""

from testw2v import runner 


if __name__ == "__main__":
    runner.main()