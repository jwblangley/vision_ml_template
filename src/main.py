import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import json

from datasets.data_loaders import load_image_folder
from evaluation.lagrangian import LagrangianMultipliers

from model.res_unet import ResUNet

from train.train import run_epoch

import utils.utils as utils

# Argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate network")

parser.add_argument(
    "-g", "--gpu", action="store_true", help="Evaluate using GPU if available"
)
parser.add_argument(
    "-d",
    "--deterministic",
    action="store_true",
    help="Set deterministic/repeatable computation",
)
parser.add_argument(
    "-t",
    "--title",
    action="store_true",
    help="Enable title prompt",
)
parser.add_argument(
    "-nl",
    "--no-log",
    action="store_true",
    help="Disable tensorboard logging. This can be useful during rapid development",
)
parser.add_argument(
    "-e", "--epochs", type=int, help="Number of epochs to train for", default=100
)
parser.add_argument("-b", "--batch-size", type=int, help="Batch size", default=8)

parser.add_argument(
    "-lr", "--learning-rate", type=float, help="Learning rate", default=1e-4
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    help="Save model_dict upon run completion",
)
parser.add_argument(
    "-cp",
    "--checkpoints",
    type=int,
    help="Epoch interval between checkpoints. No checkpoints otherwise",
)
parser.add_argument(
    "--resume",
    type=int,
    help="ID of a previous run to resume running (if specified). Resuming with modified args should be attempted with caution",
)
parser.add_argument(
    "--conditions",
    type=str,
    help="Path to JSON file containing conditional loss definitions",
)

parser.add_argument("dataset", type=str, help="Name of dataset folder within `data` directory")

args = parser.parse_args()


# Define constants
if args.gpu and not torch.cuda.is_available():
    print("[WARNING]\t GPU evaluation requested and no CUDA device found: using CPU")
DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

DETERMINISTIC = args.deterministic
if DETERMINISTIC:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(100)
    np.random.seed(42)

CHECKPOINTS = args.checkpoints
RESUME = args.resume is not None
SAVE = args.save

if RESUME:
    ID = args.resume
    print(f"Resuming ID: {ID}")
elif SAVE or CHECKPOINTS is not None:
    ID = utils.new_id()
    os.mkdir(f"out/{ID}")
    print(f"Saving with ID: {ID}")

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
ASK_TITLE = args.title
LOGGING = not args.no_log
DATASET = args.dataset
COND_LOSSES_PATH = args.conditions



# Main method
if __name__ == "__main__":
    starting_epoch = 1
    if RESUME:
        if not os.path.isdir(f"out/{ID}"):
            raise RuntimeError(f"No checkpoint folder: {ID}")
        checkpoint = utils.latest_checkpoint(ID)
        starting_epoch = checkpoint["epoch"] + 1

    # Create data loaders
    (
        training_loader,
        validation_loader,
        test_loader
    ) = load_image_folder(DATASET, BATCH_SIZE, truncate_number=200, num_workers=16)

    model = ResUNet(3, 3).to(DEVICE)

    # Set up optimsier
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Set up lagrangian multipliers if enabled
    lagrangian_multipliers = None
    if COND_LOSSES_PATH is not None:
        with open(COND_LOSSES_PATH, "r") as cond_losses_file:
            cond_losses = json.load(cond_losses_file)
        if len(cond_losses) > 0:
            lagrangian_multipliers = LagrangianMultipliers(cond_losses, device=DEVICE)

    # Set up tensorboard logging
    writer = SummaryWriter() if LOGGING else None

    if ASK_TITLE:
        writer.add_text("Title", input("Title of run: "))

    if (SAVE or CHECKPOINTS is not None) and not RESUME:
        writer.add_text("ID", f"{ID}")

    if RESUME:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        if lagrangian_multipliers is not None:
            lagrangian_multipliers.load_state_dict(checkpoint["lagrangian_multipliers"])
        torch.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    for epoch in range(starting_epoch, EPOCHS+1):
        print(f"Epoch {epoch} of {EPOCHS}")

        train_report = run_epoch(
            True,
            model,
            training_loader,
            optimizer,
            DEVICE,
            lagrangian_multipliers=lagrangian_multipliers
        )
        if LOGGING:
            for key in train_report.keys():
                utils.tensorboard_write(writer, epoch, key, train_report[key], prefix="Train")

        validation_report = run_epoch(
            False,
            model,
            validation_loader,
            optimizer,
            DEVICE,
            lagrangian_multipliers=lagrangian_multipliers
        )
        if LOGGING:
            for key in validation_report.keys():
                utils.tensorboard_write(writer, epoch, key, validation_report[key], prefix="Validation")


        # Save checkpoint if necessary
        if CHECKPOINTS is not None and epoch % CHECKPOINTS == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            }
            if lagrangian_multipliers is not None:
                checkpoint = {
                    **checkpoint,
                    **{"lagrangian_multipliers": lagrangian_multipliers.state_dict()}
                }
            torch.save(checkpoint, f"{utils.OUT_DIR}/{ID}/{epoch}-checkpoint.tar")

    writer.close()

    # Save final model
    if SAVE:
        torch.save(model.state_dict(), f"{utils.OUT_DIR}/{ID}/model.pt")
