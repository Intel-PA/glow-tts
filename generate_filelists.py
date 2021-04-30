import os
import sys
import math
import json
import random
from random import shuffle

OUT_DIR = "filelists"
SEED = 1234
EPOCHS = 100
SPLIT = {"train": 0.9542, "val": 0.0076, "test": 0.0382}


def make_dir_name(iteration: int, gamma: float, dataset_name: str) -> str:
    gamma = str(gamma).replace(".", "g")
    return f"{OUT_DIR}/{dataset_name}_{gamma}/run_{iteration}"


def sample_dataset(full_dataset: [str], gamma: float) -> list:
    resized_dataset_len = math.ceil(gamma * len(full_dataset))
    resized_dataset = []
    indices_seen = []

    while len(resized_dataset) < resized_dataset_len:
        index = random.randint(0, len(full_dataset) - 1)
        if index not in indices_seen:
            resized_dataset.append(full_dataset[index])
            indices_seen.append(index)

    return resized_dataset


def split_dataset(dataset: [str], ratios: dict) -> ([str], [str], [str]):
    shuffle(dataset)
    dataset_len = len(dataset)
    train_end_index = int(ratios["train"] * dataset_len)
    val_end_index = int((ratios["train"] + ratios["val"]) * dataset_len)
    train_list = dataset[0:train_end_index]
    val_list = dataset[train_end_index:val_end_index]
    test_list = dataset[val_end_index:]
    # print(f"[0 : {train_end_index}] [{train_end_index} : {val_end_index}] [{val_end_index} : ]")
    return train_list, val_list, test_list


def make_json(train_files: str, val_files: str, epochs: int, load_mels: bool) -> dict:
    return {
        "train": {
            "use_cuda": True,
            "log_interval": 20,
            "seed": 1234,
            "epochs": epochs,
            "learning_rate": 1e0,
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "warmup_steps": 4000,
            "scheduler": "noam",
            "batch_size": 32,
            "ddi": True,
            "fp16_run": True,
        },
        "data": {
            "load_mel_from_disk": load_mels,
            "training_files": train_files,
            "validation_files": val_files,
            "text_cleaners": ["english_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "add_noise": True,
            "cmudict_path": "data/cmu_dictionary",
        },
        "model": {
            "hidden_channels": 192,
            "filter_channels": 768,
            "filter_channels_dp": 256,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "n_blocks_dec": 12,
            "n_layers_enc": 6,
            "n_heads": 2,
            "p_dropout_dec": 0.05,
            "dilation_rate": 1,
            "kernel_size_dec": 5,
            "n_block_layers": 4,
            "n_sqz": 2,
            "prenet": True,
            "mean_only": True,
            "hidden_channels_enc": 192,
            "hidden_channels_dec": 192,
            "window_size": 4,
        },
    }


def write_files(directory: str, train: [str], val: [str], test: [str]) -> None:
    os.makedirs(directory, exist_ok=True)
    train_files = f"{directory}/train.txt"
    val_files = f"{directory}/val.txt"
    test_files = f"{directory}/test.txt"

    typical_filename = train[0].strip().split("|")[0]
    load_mel_from_disk = False if typical_filename.endswith(".wav") else True 

    with open(train_files, "w") as fh:
        for line in train:
            fh.write(line)

    with open(val_files, "w") as fh:
        for line in val:
            fh.write(line)

    with open(test_files, "w") as fh:
        for line in test:
            fh.write(line)

    config = make_json(train_files, val_files, EPOCHS, load_mel_from_disk)
    with open(f"{directory}/config.json", "w") as fh:
        json.dump(config, fh, indent=4)

if __name__ == "__main__":
    # random.seed(SEED)
    dataset_name = sys.argv[1]
    dataset_path = sys.argv[2]
    gamma = float(sys.argv[3])
    num_iterations = int(sys.argv[4])

    with open(dataset_path, "r") as fh:
        lines = fh.readlines()

    print(f"Preparing {num_iterations} subsets with gamma={gamma} from {dataset_path} .")
    for iteration in range(num_iterations):
        s = sample_dataset(lines, gamma)
        train, val, test = split_dataset(s, SPLIT)
        out = make_dir_name(iteration, gamma, dataset_name)
        write_files(out, train, val, test)
        print(f"Completed {iteration+1}/{num_iterations} runs." , end='\r', flush=True)
    