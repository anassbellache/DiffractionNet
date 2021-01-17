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
import copy

np.random.seed(0)
torch.manual_seed(0)

num_classes = 2
EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for X, y in dataset_dl:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss_b, metric_b=loss_batch(loss_func, output, y, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        if sanity_check is True:
            break
    
    loss = running_loss / float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric
    
data_dir = "./data/synthetic"
path2weights = "./models/weights.pt"
data_transformer = transforms.Compose([transforms.ToTensor()])

train_ds = DiffraNetDataset(data_dir, data_transformer)
val_ds = DiffraNetDataset(data_dir, transform=data_transformer, data_type="real_preprocessed", phase="validation")

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=True)

model = get_classifier(num_classes, device)
loss_func = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_history = {
    "train": [],
    "val": [],
}

metric_history = {
    "train": [],
    "val": [],
}
best_loss = 10

for epoch in range(EPOCHS):
    running_loss = []
    model.train()
    train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt=optimizer)
    loss_history["train"].append(train_loss)
    metric_history["train"].append(train_metric)

    model.eval()
    with torch.no_grad():
        val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

    print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
    print("-"*10)

torch.save(model.state_dict(), path2weights)

plt.title("Train-Val Loss")
plt.plot(range(1, EPOCHS+1), loss_history["train"], label="train")
plt.plot(range(1, EPOCHS+1), loss_history["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()