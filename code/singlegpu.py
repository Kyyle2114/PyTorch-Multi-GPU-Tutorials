import torch 
import torch.nn as nn 

from torch.utils.data import DataLoader

import numpy as np 
import time 

from torchinfo import summary

from data import dataset
from model import resnet50
from util import seed, trainer

def run():
    seed.seed_everything(21)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device \n')
    
    ### Train / Validation set ###    
    train_set, val_set = dataset.load_CIFAR10(root='cifar10')
    
    train_loader = DataLoader(dataset=train_set, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    ### Model ###
    model = resnet50.ResNet50(num_classes=10).to(device)
    
    print()
    print('=== MODEL INFO ===')
    summary(model)
    print()

    ### Training config ### 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    EPOCHS = 10    
    max_loss = np.inf
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = trainer.model_train(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device)
        
        train_loss = train_loss.item()
        train_acc = train_acc.item()
        
        val_loss, val_acc = trainer.model_evaluate(
            model=model, 
            data_loader=val_loader, 
            criterion=criterion, 
            device=device)

        if val_loss < max_loss:
            print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
            max_loss = val_loss
            torch.save(model.state_dict(), 'Best_Model_SINGLE.pth')

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, accuracy: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f} \n')
    
    print('=== DONE === \n')    
    
if __name__ == '__main__':
    
    DOWNLOAD_CIFAR10 = True 
    
    if DOWNLOAD_CIFAR10:
        dataset.download_cifar10()

    start_time = time.time()
    
    run()
    
    end_time = time.time()
    
    print('Elapsed time %s'%(end_time - start_time))
    
    
    