import torch
import torch.nn.functional as F


def mse_loss(predicted, target):
    """
    Computes the Mean Squared Error (MSE) loss.

    Args:
        predicted (torch.Tensor): Predicted tensor (e.g., generated images).
        target (torch.Tensor): Ground truth tensor.

    Returns:
        torch.Tensor: Scalar loss value representing the MSE.
    """
    return F.mse_loss(predicted, target)
