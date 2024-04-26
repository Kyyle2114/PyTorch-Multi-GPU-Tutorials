import torch
import numpy as np 

from tqdm import tqdm 
from typing import Tuple

def model_train(model, 
                data_loader, 
                criterion, 
                optimizer, 
                device) -> torch.Tensor:
    """
    Model train (with DDP, multiclass classification)

    Args:
        model (torch model)
        data_loader (torch dataLoader)
        criterion (torch loss)
        optimizer (torch optimizer)
        device (str)

    Returns:
        torch.Tensor: Average loss & accuracy for 1 epoch.
    """
    model.train()
    
    n_data = 0
    train_loss_ = 0.0
    train_acc_ = 0.0
    
    # tqdm set
    if (device == '0') or (device == 'cuda:0'):
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            pred = output.max(dim=1)[1]
            train_acc_ += pred.eq(y).sum()
            train_loss_ += loss * X.size(0)
            
            n_data += X.size(0)
            
    else:
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            pred = output.max(dim=1)[1]
            train_acc_ += pred.eq(y).sum()
            train_loss_ += loss * X.size(0)
            
            n_data += X.size(0)
        
    train_loss = train_loss_ / n_data
    train_acc = train_acc_ / n_data
    
    return train_loss, train_acc


def model_evaluate(model, 
                   data_loader, 
                   criterion, 
                   device) -> Tuple[float, float]:
    """
    Model validation (with multiclass classfication)

    Args:
        model (torch model)
        data_loader (torch dataloader)
        criterion (torch loss)
        device (str)

    Returns:
        Tuple[float, float]: Average loss & accuracy for 1 epoch.
    """
    model.eval()
    
    with torch.no_grad(): 
        
        val_loss_ = 0.0
        val_acc_ = 0.0

        for X, y in data_loader:
            X, y = X.to(device), y.float().to(device)   
            
            output = model(X)
            pred = output.max(dim=1)[1]
            
            val_acc_ += torch.sum(pred.eq(y)).item()
            val_loss_ += criterion(output, y).item() * X.size(0)
            
        val_acc = val_acc_ / len(data_loader.dataset)
        val_loss = val_loss_ / len(data_loader.dataset)  

        return val_loss, val_acc
    
