import logging
import numpy as np
#default logging level is warning
import torch
import torch.nn as nn
import torch.nn.functional as F
#model and data
from models.model_admin import CNN
from models.dataset import BreastCancerDataset
#validation metrics
from sklearn.metrics import confusion_matrix

#Input:
#   - Model
#   - Train dataloader
#   - Validation dataloader
#   - Loss function
#   - Optimizer
#   - Number of epochs
#   - Device
#Return:
#   - Best model
#   - Validation f1
#   - train f1
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def val_get_cm(net,test_dataloader):
    net.eval()
    cm = np.zeros((2,2))
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        cm += confusion_matrix(torch.round(outputs).detach().numpy(), labels.unsqueeze(1).detach().numpy())
    return cm

def metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel() #ravel() flatten the array
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision + recall)
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    print(f"f1: {f1}, specificity :{spec}, sensitivity {sens}")
    return f1

def train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        loss_fn='cross_entropy', 
        optimizer='adam', 
        num_epochs = 10, 
        device='cpu'
        ):

    best_model = None
    best_val_f1 = 0.0
    best_train_f1 = 0.0
    val_f1s = []
    train_f1s = []
    #Assign loss function
    if loss_fn == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn == 'nll':
        loss_fn = nn.NLLLoss()
    else:
        raise ValueError('Invalid loss function')
    #Assign optimizer
    #   - Adam: For CNN
    #   - SGD : For linear regression
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters())
    else:
        raise ValueError('Invalid optimizer')
    #Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = []
        running_acc = []
        conf_matrix = torch.zeros(2,2)
        for i, data in enumerate(train_dataloader):
            inputs,labels = data
            #set back the gradient to zero
            optimizer.zero_grad()
            #bring the input to the device
            outputs = model(inputs.to(device))
            #calculate the loss
            loss = loss_fn(outputs,labels.to(device))
            #calculate the gradient
            loss.backward()
            #update the weights
            optimizer.step()
            #Validation informations
            running_acc.append(binary_acc(outputs, labels.unsqueeze(1)))
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            conf_matrix += confusion_matrix(outputs,labels)
            running_loss.append(loss.item())

        print(f'Epoch {epoch + 1} / {num_epochs}')
        print(f'Train loss: {sum(running_loss) / len(running_loss)}')
        print(f'Train acc: {sum(running_acc) / len(running_acc)}')
        print(f'Confusion matrix: {conf_matrix}')
        train_f1 = metrics_from_cm(conf_matrix)
        train_f1s.append(train_f1)
        #Validation loop
        f1 = metrics_from_cm(val_get_cm(model,val_dataloader))
        val_f1s.append(f1)
        if f1 > best_val_f1:
            print(f'Validation f1 improved from {best_val_f1} to {f1}')
            best_val_f1 = f1
            best_model = model
            best_train_f1 = train_f1
        print("\n")
        return best_model, best_val_f1, best_train_f1, val_f1s, train_f1s



    
