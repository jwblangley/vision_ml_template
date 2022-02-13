import os

# N.B: python random should not be used for anything reproducible as it is not seeded. Use here is for ID generation
import random

import torch

import matplotlib

OUT_DIR = "./out"

def new_id():
    chosen_id = None
    while chosen_id is None or os.path.isdir(f"out/{chosen_id}"):
        chosen_id = random.randint(1e3, 1e4)
    return chosen_id


def latest_checkpoint(run_id):
    directory = f"{OUT_DIR}/{run_id}"

    max_valid = -1
    for filename in os.listdir(directory):
        if filename.endswith("-checkpoint.tar"):
            cur_id = int(filename[: -len("-checkpoint.tar")])
            max_valid = max(max_valid, cur_id)

    if max_valid <= 0:
        raise RuntimeError(f"No valid checkpoints saved for ID: {run_id}")
    return torch.load(f"{directory}/{max_valid}-checkpoint.tar")

def tensorboard_write(writer, epoch, key, value, prefix=""):
    if isinstance(value, matplotlib.figure.Figure):
        writer.add_figure(f"{prefix} {key}", value, global_step=epoch)
    elif isinstance(value, float):
        writer.add_scalar(f"{prefix} {key}", value, global_step=epoch)
    elif isinstance(value, torch.Tensor):
        if len(value.shape) == 0 or (len(value.shape) == 1 and value.size(0) == 1):
            writer.add_scalar(f"{prefix} {key}", value, global_step=epoch)
        else:
            writer.add_histogram(f"{prefix} {key}", value, global_step=epoch)
    else:
        raise RuntimeError(f"Unknown type to write to tensorboard: {type(value)}")
