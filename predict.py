import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import DiffraNetDataset
from model.deep_freak import get_classifier
from torch2trt import TRTModule, torch2trt
from torchvision import transforms, utils

np.random.seed(0)
torch.manual_seed(0)
EPOCHS = 10

num_classes = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "./data/"

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

print("loading model")
model = get_classifier(num_classes, device)
model.load_state_dict(torch.load('./model/weights.pt'))
model.eval()
print("Done!")

data_transformer = transforms.Compose([transforms.ToTensor()])
test_ds = DiffraNetDataset(data_dir, transform=data_transformer, 
                            data_type="real_preprocessed", phase="test")

print('building data loader')
test_dl = DataLoader(test_ds, batch_size=8, shuffle=True)

X, y = next(test_dl)
X = X.to(device)
y = y.to(device)

print("builfing model trt...")
model_trt = torch2trt(model, [X], fp16_mode=True)
running_metric = 0.0
len_data = len(test_dl.dataset)

def metric_epoch(model, dataset_dl, sanity_check=False):
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for X, y in dataset_dl:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        metric_b = metrics_batch(output, y) 
        running_metric += metric_b
        if sanity_check is True:
            break
    
    metric = running_metric/float(len_data)
    return metric

for epoch in range(EPOCHS):
    with torch.no_grad():
        val_metric = metric_epoch(model_trt, test_dl)
        print("epoch: %.2f accuracy: %.2f" %(epoch, 100*val_metric))
        print("-"*10)
