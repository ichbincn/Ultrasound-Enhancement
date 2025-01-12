import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, efficientnet_b0


class FeatureExtractor(nn.Module):
    """
    Feature extractor for extracting features from input images.
    Supports multiple backbones such as ResNet and EfficientNet.
    """
    def __init__(self, backbone='resnet50', pretrained=True):
        """
        Initialize the feature extractor.
        Args:
            backbone (str): The name of the backbone model ('resnet50', 'resnet18', 'efficientnet_b0').
            pretrained (bool): Whether to use a pretrained model.
        """
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            self.feature_dim = 2048  # Output dimension for ResNet50
        elif self.backbone_name == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            self.feature_dim = 512  # Output dimension for ResNet18
        elif self.backbone_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280  # Output dimension for EfficientNet-B0
        else:
            raise ValueError(f"Backbone {backbone} is not supported.")

        # Remove the classification layer (fully connected layer) from the backbone
        self._remove_fc_layer()

    def _remove_fc_layer(self):
        """
        Remove the fully connected (classification) layer from the backbone.
        """
        if 'resnet' in self.backbone_name:
            self.backbone.fc = nn.Identity()  # Replace fully connected layer with Identity
        elif 'efficientnet' in self.backbone_name:
            self.backbone.classifier = nn.Identity()  # Replace classifier with Identity

    def forward(self, x):
        """
        Forward pass to extract features.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            torch.Tensor: Extracted features (B, D), where D is the feature dimension.
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Example usage:
    model = FeatureExtractor(backbone='resnet50', pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    features = model(dummy_input)
    print("Extracted features shape:", features.shape)  # Expected shape: (4, 2048) for ResNet50
