import torch


def gradient_norm(model):
    """
    Calculate the gradient 2-norm of a model's parameters.
    This can be used to assess how much the model still has to learn.

    Parameters:
        model: model with parameters to be considered
    """
    total_norm = torch.zeros((1,))

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.pow(2).cpu()

    total_norm = total_norm.sqrt()

    return total_norm.item()
