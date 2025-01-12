import torch
import torch.nn.functional as F


def ssim_loss(predicted, target, C1=1e-4, C2=9e-4):
    """
    Computes the Structural Similarity (SSIM) loss.

    Args:
        predicted (torch.Tensor): Predicted tensor (e.g., generated images).
        target (torch.Tensor): Ground truth tensor.
        C1 (float): Stability constant for luminance comparison.
        C2 (float): Stability constant for contrast comparison.

    Returns:
        torch.Tensor: Scalar loss value representing the SSIM loss.
    """
    # Compute mean and variance for predicted and target
    mu_pred = F.avg_pool2d(predicted, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_pred = F.avg_pool2d(predicted ** 2, kernel_size=3, stride=1, padding=1) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target ** 2, kernel_size=3, stride=1, padding=1) - mu_target ** 2
    sigma_cross = F.avg_pool2d(predicted * target, kernel_size=3, stride=1, padding=1) - mu_pred * mu_target

    # Compute SSIM index
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

    # Convert SSIM to loss
    return 1 - ssim.mean()
