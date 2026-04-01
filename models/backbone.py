import torch.nn as nn
from torchvision import models

class ResNetBackbone(nn.Module):
    """
    ResNet Backbone for Feature Extraction
    Input: [B, 3, H, W]
    Output: [B, Embedding_Dim]
    """
    def __init__(self, model_name='resnet18', pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.embedding_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.embedding_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Remove the final FC layer (Step 2: Output embedding vector)
        self.model.fc = nn.Identity()

    def forward(self, x):
        # resnet18 returns [B, 512] after Identity layer
        embedding = self.model(x)
        return embedding
