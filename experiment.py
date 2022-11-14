import sys
import utils
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
from init import main as init_main
from train import main as train_main
from audio_aug.augment import get_augment_schemes


if __name__ == '__main__':
    experiment_name = "proportionExperiment"
    base_args = sys.argv[1:]  # Remove script name
    adsmote_template = "audio_aug/adsmote_scheme.yml"
    adsmote_runs = get_augment_schemes(gammas=[0.875, 0.75, 0.5, 1],
                                       num_runs=3,
                                       template_file=adsmote_template,
                                       name_prefix=experiment_name)

    for run_num, augmentor in enumerate(adsmote_runs):
        run_args = base_args.copy()
        run_args.append('-m')
        run_args.append(augmentor.config['run_name'])
        hps = utils.get_hparams(run_args)
        logging.info(f"Starting Run: {augmentor.config['run_name']} with gamma={augmentor.config['params']['gamma']}")
        init_main(hps)
        train_main(hps, augmentor, run_num)
