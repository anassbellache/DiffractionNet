import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from data.dataset import DiffraNetDataset
from model.deep_freak import get_classifier, get_neural_net
from model.custom_resnet import ResNet50, ResNet18
from torch.autograd import Variable

from statistics import mean
import copy
import optuna

np.random.seed(0)
torch.manual_seed(0)

num_classes = 2
EPOCHS = 8000
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
    corrects = Variable(pred.eq(target.view_as(pred)).sum().type(torch.DoubleTensor), requires_grad=True)
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    #print('lossfunc', loss_func)
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
        X = Variable(X, requires_grad=True).to(device)
        y = y.to(device)
        output = model(X)
        loss_b, metric_b = loss_batch(loss_func, output, y, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        if sanity_check is True:
            break
    
    loss = running_loss / float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric
    
path2weights = "./model/weights.pt"


def get_diffract_data():
    data_dir = "./data/"
    data_transformer = transforms.Compose([transforms.ToTensor()])
    train_ds = DiffraNetDataset(data_dir, transform=data_transformer, data_type="synthetic", phase="train")
    val_ds = DiffraNetDataset(data_dir, transform=data_transformer, data_type="synthetic", phase="validation")
    
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=True)
    
    return train_dl, val_dl

loss_history = {
    "train": [],
    "val": [],
}

metric_history = {
    "train": [],
    "val": [],
}
best_loss = 10

def objective(trial):
    #model = get_neural_net(trial).to(device)
    model = ResNet18(trial).to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader, valid_loader = get_diffract_data()
    
    for epoch in range(EPOCHS):
        model.train()
        
        train_loss, train_metric = loss_epoch(model, metrics_batch, train_loader, opt=optimizer)
        #loss_history["train"].append(train_loss)
        #metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, metrics_batch, valid_loader)
            #loss_history["val"].append(val_loss)
            #metric_history["val"].append(val_metric)
        
        return val_metric


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print(" Value: ", trial.value)

    print(" Params: ")
    for key, value in trial.params.items():
        print(" {}:{}".format(key, value))


""" if val_loss < best_loss:
    best_loss = val_loss
    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), path2weights)
    print("Copied best model weights!") """

#print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
#print("-"*10)

""" torch.save(model.state_dict(), path2weights)

plt.title("Train-Val Loss")
plt.plot(range(1, EPOCHS+1), loss_history["train"], label="train")
plt.plot(range(1, EPOCHS+1), loss_history["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show() """
