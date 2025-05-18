import torch.nn as nn
import torch.nn.functional as F

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# class ImageCNN(nn.Module):
#     def __init__(self):
#         super(ImageCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.fc1 = nn.Linear(128 * 16 * 16, 512)
#         self.fc2 = nn.Linear(512, 2)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x))) # [B, 32, 64, 64]
#         x = self.pool(F.relu(self.conv2(x))) # [B, 64, 32, 32]
#         x = self.pool(F.relu(self.conv3(x))) # [B, 128, 16, 16]
#         x = x.view(-1, 128 * 16 * 16)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x