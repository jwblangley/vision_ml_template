from evaluation.lagrangian import mdmm_lagrangian
from evaluation.metrics import gradient_norm
from tqdm import tqdm

import torch
import torch.nn as nn

from visualiser.visualiser import create_summary_figure

MAX_NUM_SHOW = 10


def _epoch_core(train, model, loader, optimizer, device, lagrangian_multipliers=None):
    """
    Run one epoch

    Paramaters:
        train (bool): whether to train (or to validate)
        model: model to train/validate
        loader (DataLoader): DataLoader to be used
        optimizer: optimizer to be used
        device: device to be used
        lagrangian_multipliers (LagrangianMultipliers): managing class for current lagrangian multipliers

    Returns:
        report (dict): key-value pairs for summary of the epoch
    """

    # Create totals for reporting
    loss_total = 0.0
    primary_loss_total = 0.0

    for data in tqdm(
        loader,
        total=len(loader),
        desc="Training  " if train else "Validating",
    ):
        # Define inputs and targets
        original_data = data[0].to(torch.get_default_dtype()).to(device)
        input_data = original_data
        target_data = data[0].to(torch.get_default_dtype()).to(device)

        if train:
            model.train()

            optimizer.zero_grad()
            if lagrangian_multipliers is not None:
                lagrangian_multipliers.zero_grad()

        else:
            model.eval()

        # Perform model forward pass
        output_data = model(input_data)

        # Calculate loss
        primary_loss_func = nn.MSELoss()
        primary_loss = primary_loss_func(output_data, target_data)

        # Define conditional losses for lagrangian multipliers
        avg_brightness = output_data.mean()
        cond_losses = {"average_brightness": avg_brightness}

        if lagrangian_multipliers is not None:
            loss = mdmm_lagrangian(primary_loss, lagrangian_multipliers, cond_losses)
        else:
            loss = primary_loss

        # Perform backpropagation
        if train:
            loss.backward()
            optimizer.step()

            if lagrangian_multipliers is not None:
                lagrangian_multipliers.step()

        # Total losses for reporting
        loss_total += loss.item()
        primary_loss_total += primary_loss.item()

    # Generate epoch report
    report = dict()
    report["Loss"] = loss_total / len(loader)

    if train:
        report["Gradient Norm"] = gradient_norm(model)

    if lagrangian_multipliers is not None:
        report["Primary Loss"] = primary_loss_total / len(loader)

        report_multipliers = {
            f"Lagrangian Multiplier: {name}": value.item()
            for name, (
                value,
                slack,
            ) in lagrangian_multipliers.lagrangian_multipliers.items()
        }

        report = {**report, **report_multipliers}

    report_conditional_losses = {
        f"Conditional loss: {name}": value for name, value in cond_losses.items()
    }
    report = {**report, **report_conditional_losses}

    report["Figure"] = create_summary_figure(
        original_data,
        input_data,
        target_data,
        output_data,
        min(MAX_NUM_SHOW, input_data.size(0)),
    )

    return report


def run_epoch(train, *args, **kwargs):
    if train:
        return _epoch_core(train, *args, **kwargs)

    with torch.no_grad():
        return _epoch_core(train, *args, **kwargs)
