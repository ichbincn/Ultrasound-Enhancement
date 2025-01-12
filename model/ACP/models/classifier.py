import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Feature extractor for extracting features from input images.
    Default backbone: ResNet50.
    """
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        else:
            raise ValueError(f"Backbone {backbone} is not supported.")

    def forward(self, x):
        """
        Extract features from input images.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            torch.Tensor: Extracted features (B, 2048).
        """
        return self.backbone(x)


class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) classifier for features extracted by a feature extractor.
    """
    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN classifier.
        Args:
            n_neighbors (int): Number of neighbors for KNN.
        """
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False  # Track whether the KNN model is trained

    def fit(self, features, labels):
        """
        Fit the KNN classifier.
        Args:
            features (np.ndarray): Feature vectors for training (N, D).
            labels (np.ndarray): Corresponding labels (N,).
        """
        self.knn.fit(features, labels)
        self.is_trained = True

    def predict(self, features):
        """
        Predict labels using the KNN classifier.
        Args:
            features (np.ndarray): Feature vectors for prediction (N, D).
        Returns:
            np.ndarray: Predicted labels (N,).
        """
        if not self.is_trained:
            raise ValueError("KNNClassifier must be trained before making predictions.")
        return self.knn.predict(features)

    def predict_proba(self, features):
        """
        Predict label probabilities using the KNN classifier.
        Args:
            features (np.ndarray): Feature vectors for prediction (N, D).
        Returns:
            np.ndarray: Predicted probabilities (N, C).
        """
        if not self.is_trained:
            raise ValueError("KNNClassifier must be trained before making predictions.")
        return self.knn.predict_proba(features)


class Classifier(nn.Module):
    """
    Full classifier model combining feature extractor and KNN classifier.
    """
    def __init__(self, backbone='resnet50', pretrained=True, n_neighbors=5):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.knn = KNNClassifier(n_neighbors=n_neighbors)

    def extract_features(self, x):
        """
        Extract features from input images using the feature extractor.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            np.ndarray: Extracted features (B, D).
        """
        with torch.no_grad():
            features = self.feature_extractor(x).cpu().numpy()  # Convert to NumPy for KNN
        return features

    def train_knn(self, train_loader):
        """
        Train the KNN classifier using the extracted features and labels.
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        """
        all_features = []
        all_labels = []
        for images, labels in train_loader:
            features = self.extract_features(images)
            all_features.append(features)
            all_labels.append(labels.cpu().numpy())

        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)
        self.knn.fit(all_features, all_labels)

    def predict(self, x):
        """
        Predict labels for input images using the KNN classifier.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            np.ndarray: Predicted labels (B,).
        """
        features = self.extract_features(x)
        return self.knn.predict(features)

    def predict_proba(self, x):
        """
        Predict label probabilities for input images using the KNN classifier.
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            np.ndarray: Predicted probabilities (B, C).
        """
        features = self.extract_features(x)
        return self.knn.predict_proba(features)


if __name__ == "__main__":
    # Example usage:
    from torch.utils.data import DataLoader
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor

    # Create a dummy dataset
    dataset = FakeData(transform=ToTensor(), size=100)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = Classifier(n_neighbors=5, pretrained=False)

    # Train the KNN classifier
    model.train_knn(train_loader)

    # Test the classifier
    test_images, _ = next(iter(train_loader))
    predictions = model.predict(test_images)
    print("Predicted labels:", predictions)
