import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 2


def get_neural_net(trial):
    n_conv1_stride = trial.suggest_int("n_conv1_stride", 1, 24)
    num_filters = trial.suggest_int("num_filters", 4, 128)
    n_pool2_size = trial.suggest_int("n_pool2_size", 1, 24)
    n_pool2_stride = trial.suggest_int("n_pool2_stride", 1, 4)
    model_resnet50 = models.resnet50(pretrained=False)
    num_ftrs = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model_resnet50.conv1 = nn.Conv2d(1, num_filters, kernel_size=7, stride=n_conv1_stride, padding=3, bias=False)
    model_resnet50.bn1 = nn.BatchNorm2d(num_filters)
    model_resnet50.maxpool = nn.MaxPool2d(kernel_size=n_pool2_size, stride=n_pool2_stride, padding=1, dilation=1, ceil_mode=False)

    return model_resnet50

def get_classifier(num_classes, device):
    model_resnet18 = models.resnet18(pretrained=False)
    num_ftrs = model_resnet18.fc.in_features
    model_resnet18.fc = nn.Linear(num_ftrs, num_classes)
    model_resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model_resnet18.to(device)
    return model_resnet18

def get_large_classifier(num_classes, device):
    model_resnet50 = models.resnet50(pretrained=False)
    num_ftrs = model_resnet50.in_features
    model_resnet50.fc = nn.Linear(num_ftrs, num_classes)

    model_resnet50.to(device)
    return model_resnet50
