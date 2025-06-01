import torch
import torch.nn as nn
import torch.nn.functional as F

class BengaliCharacterNet(nn.Module):
    def __init__(self, num_classes):
        super(BengaliCharacterNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the output
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def get_model(num_classes, device='cuda'):
    """
    Initialize and return the model
    """
    model = BengaliCharacterNet(num_classes=num_classes)
    model = model.to(device)
    return model
