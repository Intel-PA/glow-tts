from .train import *

CHKPT_DIR = "checkpoints/optuna_trials"
MODEL_DIR = "model/optuna_trials"
PROJECT = "glow-tts"


def setup_dirs(study_name, trial_number):
    os.makedirs(f"{CHKPT_DIR}/{study_name}/{trial_number}", exist_ok=True)

    return f"{CHKPT_DIR}/{study_name}/{trial_number}"


def hps_set_params(trial, params):
    params.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    params.n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256, 512, 1024])
    params.dropout_rate = trial.suggest_float("dropout_rate", 0, 0.75, step=0.25)
    return {
        "learning_rate": params.learning_rate, 
        "n_hidden": params.n_hidden,
        "dropout_rate": params.dropout_rate,
    }

def new_trial_callback(study, trial):
    chkpt_path = setup_dirs(study.study_name, trial.number + 1)
    FLAGS.checkpoint_dir = chkpt_path 
    FLAGS.save_checkpoint_dir = chkpt_path 
    FLAGS.load_checkpoint_dir = chkpt_path 

def objective(trial):
    params = hps_set_params(trial)
    wandb.init(project=PROJECT, config=params, reinit=True)
    val_loss = hps_train_and_eval(trial, n_gpus, hps)
    wandb.join()
    return float(val_loss)


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'

    hps = utils.get_hparams()
    study = optuna.create_study(study_name=PROJECT, direction='minimize')
    chkpt_dir = setup_dirs(study.study_name, 0)
    study.optimize(objective, n_trials=25, callbacks=[new_trial_callback])

if __name__ == "__main__":
    main()

