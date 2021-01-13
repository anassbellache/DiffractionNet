import os
import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from data.dataset import DiffraNetDataset
from model.deep_freak import get_classifier

from statistics import mean

def show(img, y=None, color=True):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    plt.imshow(npimg_tr)
    if y is not None:
        plt.title("label: "+str(y))
    plt.show()

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return loss.item(), metric_b



np.random.seed(0)
torch.manual_seed(0)

num_classes = 5
EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "./data/synthetic"
data_transformer = transforms.Compose([transforms.ToTensor()])

train_ds = DiffraNetDataset(data_dir, data_transformer)

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

model = get_classifier(num_classes, device)
loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(EPOCHS):
    running_loss = []
    for imgs, labels in train_dl:
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)
        loss_b, metric_b = loss_batch(loss_func, output, labels, opt)
        running_loss.append(loss_b)
    
    if epoch % 10 == 0:
        print('epoch: {}  loss: {}'.format(epoch + 1, mean(running_loss)))
