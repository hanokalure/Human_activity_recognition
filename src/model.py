import torch.nn as nn
from torchvision import models

def get_model(num_classes=101):  # 👈 UCF101 has 101 classes
    model = models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
