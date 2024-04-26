import torch
import torch.nn as nn 

from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        """
        Pytorch ResNet50 implementation.
        Using ImageNet weights.

        Args:
            num_classes (int, optional): number of classes. Defaults to 10(CIFAR-10).
        """
        
        super(ResNet50, self).__init__()
        
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights, progress=False)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        for _, p in self.model.named_parameters():
            p.requires_grad = False 
            
        for _, p in self.model.fc.named_parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)