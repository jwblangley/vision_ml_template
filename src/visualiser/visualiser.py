import warnings
import numpy as np

import torch

from torchvision.utils import make_grid

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

@torch.no_grad()
def create_summary_figure(
    original_data,
    input_data,
    target_data,
    output_data,
    num_show,
    title=None,
):
    """
    Create visualised distribution figure

    Parameters:
        input_data (BxCxHxW Tensor): input data
        target_data (BxCxHxW Tensor): input data
        output_data (BxCxHxW Tensor): output data
        num_show (int): number of data points to show in figure (must be less than or equal to batch size)
        title (string): title of figure

    Returns:
        matplotlib.figure.Figure
    """

    batch_size = output_data.size(0)

    if num_show > batch_size:
        raise RuntimeError(
            "Asking to display more images than available in a single batch"
        )

    if not (input_data.shape == target_data.shape == output_data.shape):
        raise ValueError(f"Mismatched data shapes. Input, Target, Output: {input_data.shape}, {target_data.shape}, {output_data.shape}")

    # Define figure parameters
    plt.clf()

    # Definte total number of plots
    total_plots = 5

    plt_counter = 0

    plt.figure(figsize=(num_show, total_plots))
    if title is not None:
        plt.suptitle(title)
    plt.subplots_adjust(
        left=0.01, right=0.98, top=0.933, bottom=0.01, hspace=0.05, wspace=0
    )

    # Calculations
    residual = output_data - target_data

    # Define rows

    # Original data
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    original_plt = original_data.clamp(0, 1)
    original_plt = make_grid(original_plt, nrow=num_show)
    original_plt = original_plt.cpu().numpy()
    original_plt = np.transpose(original_plt, (1, 2, 0))

    plt.ylabel("Original")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(original_plt)

    # Input data
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    input_plt = input_data.clamp(0, 1)
    input_plt = make_grid(input_plt, nrow=num_show)
    input_plt = input_plt.cpu().numpy()
    input_plt = np.transpose(input_plt, (1, 2, 0))

    plt.ylabel("Input")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(input_plt)

    # Target data
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    target_plt = target_data.clamp(0, 1)
    target_plt = make_grid(target_plt, nrow=num_show)
    target_plt = target_plt.cpu().numpy()
    target_plt = np.transpose(target_plt, (1, 2, 0))

    plt.ylabel("Target")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(target_plt)

    # Output data
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    output_plt = output_data.clamp(0, 1)
    output_plt = make_grid(output_plt, nrow=num_show)
    output_plt = output_plt.cpu().numpy()
    output_plt = np.transpose(output_plt, (1, 2, 0))

    plt.ylabel("Output")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(output_plt)

    # Residual
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    residual_plt = make_grid(residual, nrow=num_show)
    residual_plt = residual_plt.cpu().numpy()
    residual_plt = np.transpose(residual_plt, (1, 2, 0))
    # To preserve scaling, show pixelwise average and remove colour dim
    residual_plt = residual_plt.mean(axis=2)

    plt.ylabel("Residual")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(residual_plt)


    assert plt_counter == total_plots, f"Incorrect final plot counter. Got {plt_counter}. Expected {total_plots}"
    return plt.gcf()
