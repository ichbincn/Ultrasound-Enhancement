import torch
import torch.nn.functional as F


def mse_loss(predicted, target):
    """
    Computes the Mean Squared Error (MSE) loss.
    Args:
        predicted (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
    Returns:
        torch.Tensor: MSE loss value.
    """
    return F.mse_loss(predicted, target)


def ssim_loss(predicted, target, C1=1e-4, C2=9e-4):
    """
    Computes the Structural Similarity (SSIM) loss.
    Args:
        predicted (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        C1 (float): Stability constant for luminance comparison.
        C2 (float): Stability constant for contrast comparison.
    Returns:
        torch.Tensor: SSIM loss value.
    """
    mu_pred = F.avg_pool2d(predicted, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_pred = F.avg_pool2d(predicted ** 2, kernel_size=3, stride=1, padding=1) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target ** 2, kernel_size=3, stride=1, padding=1) - mu_target ** 2
    sigma_cross = F.avg_pool2d(predicted * target, kernel_size=3, stride=1, padding=1) - mu_pred * mu_target

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    return 1 - ssim.mean()


def combined_loss(predicted, target, lambda_ssim=0.84):
    """
    Combines MSE loss and SSIM loss for image reconstruction tasks.
    Args:
        predicted (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        lambda_ssim (float): Weight for SSIM loss.
    Returns:
        torch.Tensor: Combined loss value.
    """
    # Compute MSE loss
    mse = mse_loss(predicted, target)

    # Compute SSIM loss
    ssim = ssim_loss(predicted, target)

    # Combine the two losses
    total_loss = mse + lambda_ssim * ssim
    return total_loss
