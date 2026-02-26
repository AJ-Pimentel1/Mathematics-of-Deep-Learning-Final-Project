import torch
import torch.nn as nn
from torchvision.models import resnet18

class NCResNet18(nn.Module):
    def __init__(self, dataset_name='cifar10', num_classes=10):
        super(NCResNet18, self).__init__()
        
        # Determine input channels based on the dataset
        if dataset_name in ['mnist', 'fmnist']:
            in_channels = 1
        elif dataset_name == 'cifar10':
            in_channels = 3
        else:
            raise ValueError("Dataset must be mnist, fmnist, or cifar10")

        # Load the base ResNet18 architecture
        self.backbone = resnet18(weights=None)
        
        # Replace the 7x7 stride-2 convolution with a 3x3 stride-1 convolution
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove the MaxPool layer entirely by replacing it with an Identity layer
        self.backbone.maxpool = nn.Identity()
        
        # ---------------------------------------------------------
        # THE CLASSIFIER (W and b)
        # ---------------------------------------------------------
        # ResNet18's final feature dimension is 512.
        self.backbone.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x, return_features=False):
        """
        By setting return_features=True, we can extract h(x) to measure Neural Collapse.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1) # This is h(x), the last-layer activations
        
        logits = self.backbone.fc(features) # This is Wh(x) + b
        
        if return_features:
            return logits, features
        return logits

# --- Example Instantiation ---
# model_cifar = NCResNet18(dataset_name='cifar10')
# model_mnist = NCResNet18(dataset_name='mnist')