import torch
import torch.nn as nn
import optuna


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_filters, layers, image_channels, num_classes, strides):
        super(ResNet, self).__init__()
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(image_channels, num_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=strides[0])
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=strides[1])
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=strides[2])
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.num_filters != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_filters, 
                    intermediate_channels*4, 
                    kernel_size=1, 
                    stride=stride),
                    nn.BatchNorm2d(intermediate_channels*4))
        layers.append(block(self.num_filters, intermediate_channels, identity_downsample, stride))
        self.num_filters = intermediate_channels*4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.num_filters, intermediate_channels))
        
        return nn.Sequential(*layers)

    
def ResNet18(trial):
    stride1 = trial.suggest_int("stride1", 1, 3)
    stride2 = trial.suggest_int("stride2", 1, 3)
    stride3 = trial.suggest_int("stride3", 1, 3)
    stride4 = trial.suggest_int("stride4", 1, 3)
    strdes = [stride1, stride2, stride3, stride4]


    num_filters = trial.suggest_int("num_filters", 1, 80)
    return ResNet(block, num_filters, [2, 2, 2, 2], 1, 2, strdes)
    
    
def ResNet50(trial):
    stride1 = trial.suggest_int("stride1", 1, 4)
    stride2 = trial.suggest_int("stride2", 1, 4)
    stride3 = trial.suggest_int("stride3", 1, 4)
    stride4 = trial.suggest_int("stride4", 1, 4)
    strdes = [stride1, stride2, stride3, stride4]


    num_filters = trial.suggest_int("num_filters", 1, 80)
    return ResNet(block, num_filters, [3, 4, 6, 3], 1, 2, strdes)

#def ResNet101(img_channels=3, num_classes=1000):
#    return ResNet(block, 64, [3, 4, 23, 3], img_channels, num_classes)



