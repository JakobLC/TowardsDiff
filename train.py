

import sys, os
sys.path.append(os.path.abspath('./source/'))

from source.utils.argparsing import TieredParser, save_args, load_existing_args
from source.training import DiffusionModelTrainer


def main(**modified_args):
    args = TieredParser().get_args(modified_args=modified_args)
    if isinstance(args,list):
        modified_args_list = args
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()
    
if __name__ == "__main__":
    main()