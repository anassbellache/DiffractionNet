import torch
from torch import nn
from torchvision import models



def get_classifier(num_classes, device):
    model_resnet18 = models.resnet18(pretrained=False)
    num_ftrs = model_resnet18.fc.in_features
    model_resnet18.fc = nn.Linear(num_ftrs, num_classes)
    model_resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model_resnet18.to(device)
    return model_resnet18

