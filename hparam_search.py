import optuna

from train import *

MODEL_DIR = "models/optuna_trials"
PROJECT = "glow-tts"


def setup_dirs(study_name, trial_number):
    os.makedirs(f"{MODEL_DIR}/{study_name}/{trial_number}", exist_ok=True)

    return f"{MODEL_DIR}/{study_name}/{trial_number}"


def hps_set_params(trial, params):
    params.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e0, log=True)
    params.p_dropout = trial.suggest_float("p_dropout", 0, 0.75, step=0.25)
    return {
        "learning_rate": params.learning_rate,
        "p_dropout": params.p_dropout,
    }

def new_trial_callback(study, trial):
    model_path = setup_dirs(study.study_name, trial.number + 1)
    FLAGS.checkpoint_dir = chkpt_path 
    FLAGS.save_checkpoint_dir = chkpt_path 
    FLAGS.load_checkpoint_dir = chkpt_path 

def objective(study, trial):
    hps = utils.get_hparams()
    model_dir = setup_dirs(study.study_name, trial.number)
    hps.epochs = 1 #delete this line
    hps.model_dir = model_dir
    params = hps_set_params(trial, hps)
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

    study = optuna.create_study(study_name=PROJECT, direction='minimize')
    study.optimize(objective, n_trials=10, callbacks=[new_trial_callback])

if __name__ == "__main__":
    main()

