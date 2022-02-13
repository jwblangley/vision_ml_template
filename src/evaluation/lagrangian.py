import torch


class LagrangianMultipliers():
    """
    Class for managing a collection of lagrangian multipliers
    """

    def __init__(self, name_slack_dict, device="cpu"):
        """
        Constructs LagrangianMultipliers class

        Parameters:
            name_slack_dict: dictionary of {name: slack_value}:
        """

        self.lagrangian_multipliers = {}

        for name, slack in name_slack_dict.items():
            if not isinstance(name, str):
                raise ValueError(f"Invalid name type. Expected str. Got {type(name)}")
            if not isinstance(slack, float):
                raise ValueError(f"Invalid slack type. Expected float. Got {type(slack)}")

            self.lagrangian_multipliers[name] = (torch.zeros(1, requires_grad=True, device=device), slack)

        self.optimizer = torch.optim.SGD([multiplier for multiplier, slack in self.lagrangian_multipliers.values()], lr=1.0)

    def __len__(self):
        return len(self.lagrangian_multipliers)

    def __getitem__(self, key):
        return self.lagrangian_multipliers[key]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # Flip gradients for gradient ascent
        for multiplier, slack in self.lagrangian_multipliers.values():
            multiplier.grad.neg_()

        # Take gradient ascent step
        self.optimizer.step()

        # Clamp multipliers into positive region
        for multiplier, slack in self.lagrangian_multipliers.values():
            if multiplier < 0:
                multiplier.data = multiplier.data * 0.0

    def state_dict(self):
        return {
            "lagrangian_multipliers": self.lagrangian_multipliers,
            "optimizer_state": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.lagrangian_multipliers = state_dict["lagrangian_multipliers"]
        self.optimizer.load_state_dict(state_dict["optimizer_state"])


def mdmm_lagrangian(primary_loss, lagrangian_multipliers, cond_losses, damping=10.0):
    """
    Calculates a Modified Differential Method of Multipliers loss

    Parameters:
        primary_loss (Tensor): loss to be optimised when all conditions are met
        lagrangian_multipliers (LagrangianMultipliers): relevant lagrangian multipliers class
        cond_losses (dict): individual losses for each multiplier

    Returns:
        result (singleton Tensor)
    """
    if not isinstance(lagrangian_multipliers, LagrangianMultipliers):
        raise ValueError(f"Invalid type for lagrangian_multipliers. Expected LagrangianMultipliers. Got {type(lagrangian_multipliers)}")

    if not isinstance(cond_losses, dict):
        raise ValueError(f"Invalid type for cond_losses. Expected dict. Got {type(cond_losses)}")
    if len(cond_losses) != len(lagrangian_multipliers):
        raise ValueError(f"Mismatched lagrangian_multipliers and cond_losses length. Expected {len(lagrangian_multipliers)}={len(lagrangian_multipliers)}. Got {len(lagrangian_multipliers)}={len(cond_losses)}")

    constraint_loss = torch.zeros_like(primary_loss, requires_grad=True)

    for name, loss in cond_losses.items():
        multiplier, slack = lagrangian_multipliers[name]

        damp = damping * (slack - loss).detach()
        constraint_loss = constraint_loss - (multiplier - damp) * (slack - loss)


    return primary_loss + constraint_loss
