import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableInputNet(nn.Module):
    def __init__(self):
        super(VariableInputNet, self).__init__()
        
        # Increase the capacity of the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1280, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)  # Additional layer

        # Batch Normalization after each conv layer
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048)

        # Adaptive Pooling layer to ensure output width is 128 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)
        
        # Linear layer to match the final dimension to 512, consider increasing size if needed
        self.fc1 = nn.Linear(2048, 1024)  # New intermediate layer
        self.fc2 = nn.Linear(1024, 486)   # Final output layer

    def forward(self, x, mask_token):
        mask_token = mask_token[0].unsqueeze(2)  # Prepare mask token
        x = x + mask_token  # Apply mask token to x
        
        x = x.transpose(1, 2)  # Transpose for convolution compatibility

        # Convolution layers with GELU activation and batch normalization
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))  # Additional conv layer
        
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = x.transpose(1, 2)  # Transpose back for linear layers

        # Linear transformations with GELU activation
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))

        return x