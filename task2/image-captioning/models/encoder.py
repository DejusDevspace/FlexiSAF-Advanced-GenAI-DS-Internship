import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    The encoder uses a pre-trained CNN (ResNet) to extract image features.
    The CNN converts images into fixed-size feature vectors.
    """
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        # Load pre-trained ResNet50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove last (fully connected) layer
        layers = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*layers)

        # Linear layer to transform ResNet output to embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """
        Extract feature vectors from images.
        Args:
            images: batch of images (batch_size, 3, 224, 224)
        Returns:
            features: embedded features (batch_size, embed_size)
        """
        # No gradient for pre-trained CNN
        with torch.no_grad():
            features = self.resnet(images)

        # Flatten
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features
