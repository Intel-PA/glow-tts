import os
import re
import sys
import json
import math
import glob
import joblib
import argparse

import torch
import optuna 
import wandb
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from utils import HParams
from data_utils import TextMelLoader, TextMelCollate
import models
import commons
import utils
from text.symbols import symbols
                            

global_step = 0

N_GPUS = None
RANK = 0
MODEL_DIR = "models/optuna_trials"
KEEP_EVERY = 20
CHKPT_PATT = r"G_\d+\.pth"
DATADIR = "/home/kjayathunge/datasets/LJS"


def start_search(gamma, aug_method, opt_config, resume=False):
    gamma = gamma_to_str(gamma)
    project = f"glow-tts_{aug_method}_{gamma}"

    if resume:
        study = joblib.load(resume)
        print(f"Best trial until now: {study.best_trial.value}")
        print(f"Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        study = optuna.create_study(study_name=project, direction='minimize', pruner=optuna.pruners.HyperbandPruner())
    
    study.optimize(lambda trial: objective(trial, study, gamma, aug_method, project, opt_config))


def objective(trial, study, gamma, aug_method, project, opt_config):
    global global_step
    global_step = 0

    filelist_dir = opt_config["filelist_dir"]
    aug = get_aug(opt_config["augs"], aug_method)
    opt_params = aug["params"]
    hps = utils.get_hparams()
    trial_params, all_params = hps_set_params(trial, project, gamma, hps, filelist_dir, aug_method, opt_params)
    
    joblib.dump(study, f"{all_params.model_dir}/study.pkl")
    wandb.init(project=project, config=trial_params, reinit=True)

    try:
        train_loss, val_loss = train_and_eval(RANK, N_GPUS, all_params, trial)
    except RuntimeError as e:
        val_loss = -1
        print(f"Trial {trial.number} encountered a runtime error: {e}")
        pass
    wandb.join()
    cleanup_dir(all_params.model_dir, KEEP_EVERY)
    return float(val_loss)


def get_model_dir_name(trial_number, project):
    return f"{MODEL_DIR}/{project}/{trial_number}"


def get_aug(augs, aug_name):
    for aug in augs:
        if aug["name"] == aug_name:
            return aug
    return None


def gamma_to_str(gamma):
    return str(gamma).replace(".", "g")


def create_symlinks(config):
    augs = config["augs"]
    for aug in augs:
        if aug["name"] == "sox":
            mus = aug["params"]["mu"]["values"]
            sigmas = aug["params"]["sigma"]["values"]
            speeds = aug["params"]["speed"]["values"]

            if not os.path.exists("DATASETS"):
                os.mkdir("DATASETS")
            for m in mus:
                for s in sigmas:
                    for v in speeds:
                        real_link = f"{DATADIR}/SOX-M{m}-S{s}-V{v}"
                        sym_link = f"DATASETS/SOX-M{m}-S{s}-V{v}"
                        if not os.path.exists(sym_link):
                            os.symlink(real_link, sym_link)


def setup_dirs(trial_number, project):
    model_dir = get_model_dir_name(trial_number, project)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def cleanup_dir(directory, interval):
    checkpoints = [f for f in os.listdir(directory) if re.match(CHKPT_PATT, f)]
    num_epochs = len(checkpoints)
    to_keep = [f"G_{x}.pth" for x in range(0, num_epochs+1, interval)][1:]
    
    for chkpt in checkpoints:
        if chkpt not in to_keep:
            os.remove(f"{directory}/{chkpt}")


def optuna_suggest(trial, name, config):
    if config["type"] == "categorical":
        values = config["values"]
        return trial.suggest_categorical(name, values)
    
    if config["type"] == "int": 
        start = int(config["start"])
        end = int(config["end"])
        step = int(config.get("step", 1))
        return trial.suggest_int(name, start, end, step)
    
    if config["type"] == "float":
        start = float(config["start"])
        end = float(config["end"])
        step = config.get("step", None)
        step = float(step) if step is not None else None
        log = config.get("log", False)
        return trial.suggest_float(name, start, end, log=log, step=step)


def hps_set_params(trial, project, gamma, params, filelist_dir, aug_method, opt_params):
    trial_params = {}
    model_dir = setup_dirs(trial.number, project)

    for param_name, config in opt_params.items():
        trial_params[param_name] = optuna_suggest(trial, param_name, config)

    config_file = f"{filelist_dir}/{gamma}/{aug_method.upper()}-M{trial_params['mu']}-S{trial_params['sigma']}-V{trial_params['speed']}/config.json"
    with open(config_file) as fh:
        params_dict = json.load(fh)
        all_params = HParams(**params_dict)

    all_params.train.batch_size = 256
    all_params.model_dir = model_dir
    # hps.train.epochs =  100 #delete this line

    # params.train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e0, log=True)
    # params.model.p_dropout = trial.suggest_float("p_dropout", 0, 0.25, step=0.05)
    return trial_params, all_params


def train_and_eval(rank, n_gpus, hps, trial):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextMelLoader(hps.data.training_files, hps.data)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextMelCollate(1)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
  if rank == 0:
    val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.val_batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

  generator = models.FlowGenerator(
      n_vocab=len(symbols) + getattr(hps.data, "add_blank", False), 
      out_channels=hps.data.n_mel_channels, 
      **hps.model).cuda(rank)
  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  if hps.train.fp16_run:
    generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
  generator = DDP(generator)
  epoch_str = 1
  global_step = 0
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
    epoch_str += 1
    optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
    optimizer_g._update_learning_rate()
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
  
  train_loss = 0
  eval_loss = 0
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_loss = train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer)
      eval_loss = evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
      utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
      print(f"val_loss: {eval_loss}, train_loss: {train_loss}")
      wandb.log({"val_loss": eval_loss, "train_loss": train_loss}, step=epoch)
      trial.report(eval_loss, epoch)
      if trial.should_prune():
            raise optuna.TrialPruned()
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)

  return train_loss, eval_loss


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
  train_loader.sampler.set_epoch(epoch)
  global global_step

  final_loss = 0
  generator.train()
  for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    # Train Generator
    optimizer_g.zero_grad()
    
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, gen=False)
    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_length = commons.duration_loss(logw, logw_, x_lengths)

    loss_gs = [l_mle, l_length]
    loss_g = sum(loss_gs)

    if hps.train.fp16_run:
      with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
        scaled_loss.backward()
      grad_norm = commons.clip_grad_value_(amp.master_params(optimizer_g._optim), 5)
    else:
      loss_g.backward()
      grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
    optimizer_g.step()
    
    if rank==0:
      if batch_idx % hps.train.log_interval == 0:
        (y_gen, *_), *_ = generator.module(x[:1], x_lengths[:1], gen=True)
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss_g.item()))
        logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])
        
        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), 
            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), 
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)
    global_step += 1
    final_loss = loss_g.item()
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

  return final_loss
 

def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    final_loss = 0
    print(f"GOT {len(val_loader.dataset)} val items.")
    with torch.no_grad():
      for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        
        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        # if batch_idx % hps.train.log_interval == 0:
        #   logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(x), len(val_loader.dataset),
        #     100. * batch_idx / len(val_loader),
        #     loss_g.item()))
        #   logger.info([x.item() for x in loss_gs])
        logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(val_loader.dataset),
          100. * batch_idx / len(val_loader),
          loss_g.item()))
        logger.info([x.item() for x in loss_gs])
        final_loss = loss_g.item()
           
    
    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))
    return final_loss
                           
if __name__ == "__main__":
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    N_GPUS = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'

    dist.init_process_group(backend='nccl', init_method='env://', world_size=N_GPUS, rank=RANK)
    
    config_path = "/home/kjayathunge/audio-augmentation/configs/glow-tts.json"
    # config_path = sys.argv[1]
    with open(config_path, "r") as fh:
        config = json.load(fh)
    create_symlinks(config)

    gammas = config["gammas"]
    resumes = [
        False, 
        False,
        False
    ]
    for gamma, resume in zip(gammas, resumes):
        print(f"Starting study for gamma={gamma}")
        try:
            start_search(gamma, "sox", config, resume)
        except KeyboardInterrupt:
            print(f"Study for gamma={gamma} cancelled.")
            pass
