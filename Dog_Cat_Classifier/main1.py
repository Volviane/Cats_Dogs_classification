#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:03:48 2020

@author: Saphir Volviane
"""

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets ,models,transforms
from model import  CNN1
from  dataset import DatasetLoader



# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np




def train(epoch, model,perm=torch.arange(0, 50176).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #we chage the device if the GPU is available
        if  torch.cuda.is_available():
              data, target = data.cuda(), target.cuda()
        # permute pixels
#         data = data.view(-1, 224*224)
#         data = data[:, perm]
#         data = data.view(-1, 3, 224, 224)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, perm=torch.arange(0, 50176).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
          #we chage the device if the GPU is available
          if  torch.cuda.is_available():
              data, target = data.cuda(), target.cuda()
          
          # permute pixels
          data = data.view(-1, 224*224)
          data = data[:, perm]
          data = data.view(-1, 3, 224, 224)
          output = model(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
          pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
          correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    






if __name__ == "__main__":
    
    
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    input_size =224*224*3
    output_size = 2
    accuracy_list = []
    # define training, valid and test data directories
    data_dir = './Cat_Dog_data/'
    train_dir = os.path.join(data_dir, 'train/')
    #valid_dir = os.path.join(data_dir, 'valid/')
    test_dir = os.path.join(data_dir, 'test/')

    ## Write data loaders for training,  and test sets
    ## Specify appropriate transforms, and batch_sizes
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                          transforms.Grayscale(),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
                                                              
    
    test_transform = transforms.Compose([transforms.Resize(size=(224,224)),
                                         transforms.Grayscale(),
                                        transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x.repeat(3,1,1)),#])
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
        
    
    train_data = DatasetLoader(train_dir, transform=train_transform)
    test_data = DatasetLoader(test_dir, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        num_workers=num_workers, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)
    
    
    # Training settings  for the first model
    n_features =8 # number of feature maps
    model_cnn = CNN1(input_size, n_features, output_size)
    #we change the device if the GPU is available
    if  torch.cuda.is_available():
      model_cnn = model_cnn.cuda()
    optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
    print('Number of parameters for the first CNN: {}'.format(get_n_params(model_cnn)))
    
    for epoch in range(0, 1):
        train(epoch, model_cnn)
        test(model_cnn)
