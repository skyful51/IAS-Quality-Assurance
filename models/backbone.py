import torch
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

class CutPasteBackbone(nn.Module):
    """
    CutPaste Backbone for Feature Extraction (ResNet18 + Projection Head MLP)
    Input: [B, 3, H, W]
    Output: [B, Embedding_Dim]
    
    This class excludes the final classification layer (self.out) from
    the CutPaste ProjectionNet, returning the representation/embedding
    to be used as the backbone of ArcMarginProduct.
    """
    def __init__(self, pretrained=True, head_layers=None, include_head=True, checkpoint_path=None):
        super(CutPasteBackbone, self).__init__()
        self.include_head = include_head
        
        if head_layers is None:
            self.head_layers = [512, 512, 512, 512, 512, 512, 512, 512, 128]
        elif isinstance(head_layers, int):
            self.head_layers = [512] * head_layers + [128]
        else:
            self.head_layers = head_layers
            
        # Base ResNet18
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Identity()
        
        if self.include_head:
            # Construct the MLP projection head matching the ProjectionNet definition
            last_layer = 512
            sequential_layers = []
            for num_neurons in self.head_layers:
                sequential_layers.append(nn.Linear(last_layer, num_neurons))
                sequential_layers.append(nn.BatchNorm1d(num_neurons))
                sequential_layers.append(nn.ReLU(inplace=True))
                last_layer = num_neurons
                
            self.head = nn.Sequential(*sequential_layers)
            self.embedding_dim = last_layer
        else:
            self.embedding_dim = 512
            
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Loads state dict from a trained CutPaste ProjectionNet checkpoint.
        Handles keys dynamically depending on whether include_head is True or False,
        and filters out the classification head weights ('out.').
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if checkpoint comes from ProjectionNet (which uses 'resnet18.' prefix for the ResNet part)
        is_projection_net = any(k.startswith('resnet18.') for k in state_dict.keys())
        
        if is_projection_net:
            if self.include_head:
                # Remove classification head weights ('out.')
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('out.')}
                self.load_state_dict(filtered_state_dict, strict=True)
            else:
                # Load only resnet18 weights and strip the prefix
                resnet_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('resnet18.'):
                        new_key = k.replace('resnet18.', '')
                        resnet_state_dict[new_key] = v
                self.resnet18.load_state_dict(resnet_state_dict, strict=True)
        else:
            # If checkpoint has standard ResNet weights, load to resnet18
            # Or if it's a standard backbone state dict, load with best effort
            has_head_keys = any(k.startswith('head.') for k in state_dict.keys())
            if has_head_keys and self.include_head:
                self.load_state_dict(state_dict, strict=False)
            else:
                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        clean_state_dict[k.replace('model.', '')] = v
                    else:
                        clean_state_dict[k] = v
                self.resnet18.load_state_dict(clean_state_dict, strict=False)

    def forward(self, x):
        embedding = self.resnet18(x)
        if self.include_head:
            embedding = self.head(embedding)
        return embedding

    def freeze_resnet(self):
        # Freeze full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        # Unfreeze fc/identity (for API compatibility, though fc is Identity)
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        # Unfreeze all
        for param in self.parameters():
            param.requires_grad = True

