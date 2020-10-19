#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:31:29 2020

@author: rakshitbhatt98
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from Main_Model import Model
import wandb


wandb.init(project = "fuse_network")

transform_train = transforms.Compose([                                                                                                                                                  
                       transforms.RandomCrop(32, padding=4),                                                                                                                               
                       transforms.RandomHorizontalFlip(),                                                                                                                                  
                       transforms.ToTensor(),                                                                                                                                              
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                                                                                           
                       ])                                                                                                                                                                  
transform_test = transforms.Compose([                                                                                                                                                   
                    transforms.ToTensor(),                                                                                                                                              
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                                                                                           
]) 


'train test dataset'

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

cuda = torch.device("cuda:0")
model = Model()
model.cuda()

wandb.watch(model)

'Loss function definition'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.cuda()
        inputs = inputs.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.forward2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            wandb.log({"Running loss after 2k batches": running_loss / 2000})
            epoch_loss = running_loss / 2000

            running_loss = 0.0
            
    wandb.log({"Loss_after_Every_epoch": epoch_loss })

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model.forward2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        wandb.log({"Test_Accuracy": 100*correct/total})
        


