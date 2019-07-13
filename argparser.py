import argparse


def str2bool(v):
    """robust boolean function for command line arg"""
    return v.lower() not in ('no', 'false', 'f', '0', False)


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parser for input arguments')
        self.parser.register('type', 'bool', str2bool)
        self.add_args()

    def add(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def parse(self):
        args = self.parser.parse_args()
        # add assertions of any of the arguments
        assert (args.valid_split < 1) and (args.valid_split >= 0), "valid_split should be within [0, 1)."
        assert args.cross_valid >= 1, "Number of cross validation should be larger than 1."
        # assert args.n_gpu_use == 1, "Keep n_gpu_use==1 for now, see https://github.com/pytorch/pytorch/issues/7890."
        return args

    def add_args(self):
        self.add('-pd', '--path_data',
                 help="The path to the dataset",
                 type=str, required=True)
        self.add('-sd', '--save_dir',
                 help="The directory to save results",
                 type=str, required=True)
        self.add('-vs', '--valid_split',
                 help="The percentage of the dataset used for validation",
                 type=float, default=0)
        self.add('-cv', '--cross_valid',
                 help="Number of cross validation. Ignore -vs if -cv > 1",
                 type=int, default=1)
        self.add('-db', '--debug',
                 type=str2bool, default=False)
        # model
        self.add('-ni', '--normalize_input', type=str2bool, default=True)
        self.add('-fd', '--f_dim', type=int, default=256)
        self.add('-zd', '--z_dim', type=int, default=32)
        self.add('-nc', '--n_channel', type=int, default=256)
        self.add('-fac', '--factorized', type=str2bool, default=True)
        # trainer
        self.add('-bs', '--batch_size', type=int, default=128)
        self.add('-ep', '--epochs', type=int, default=500)
        self.add('-sp', '--save_period', type=int, default=10)
        self.add('-es', '--early_stop', type=int, default=100)
        self.add('-ngu', '--n_gpu_use', type=int, default=1)

        self.add('-res', '--resume',
                 help="The model checkpoint to continue training. All arguments will be ignored if provided.",
                 type=str, default=None)
