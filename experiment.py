import sys
import utils
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
from init import main as init_main
from train import main as train_main
from audio_aug.augment import get_augment_schemes


if __name__ == '__main__':
    augment_runs = get_augment_schemes(gammas=[0.875, 1],
                                       num_runs=2,
                                       template_file="audio_aug/adsmote_scheme.yml",
                                       name_prefix="experiment1")

    base_args = sys.argv[1:]  # Remove script name
    for augmentor in augment_runs:
        run_args = base_args.copy()
        run_args.append('-m')
        run_args.append(f"{augmentor.config['run_name']}_g{augmentor.config['params']['gamma']}")
        hps = utils.get_hparams(run_args)
        logging.info(f"Starting Run: {augmentor.config['run_name']} with gamma={augmentor.config['params']['gamma']}")
        init_main(hps)
        train_main(hps, augmentor)
