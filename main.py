import numpy as np
from classes.FileLoader import FileLoader
import sys
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == '__main__':
    img_dir = 'cell_images/'
    print(os.listdir(img_dir))

    # Transforms
    train_transforms = transforms.Compose(
            [transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(),
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
            
    validation_transforms = transforms.Compose(
            [transforms.Resize(256),
            transforms.ToTensor(), 
            transforms.CenterCrop(224),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    test_transforms = transforms.Compose(
            [transforms.Resize(256),
            transforms.ToTensor(), 
            transforms.CenterCrop(224),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    # Datasets
    train_data = datasets.ImageFolder(img_dir, transform=train_transforms)

    num_workers = 0

    validation_size = 0.2

    test_size = 0.1

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    validation_split = int(np.floor( (validation_size * num_train )))
    test_split = int(np.floor( (validation_size + test_size) * num_train ))
    validation_idx, test_idx, train_idx = indices[:validation_split], indices[validation_split:test_split], indices[test_split:]

    print('Train base      => ', len(train_idx))
    print('Validation base => ', len(validation_idx))
    print('Test base       => ', len(test_idx))

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=validation_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=20, sampler=test_sampler, num_workers=num_workers)

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2, bias=True)

    fc_parameters = model.fc.parameters()

    for param in fc_parameters:
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    valid_loss_min = np.Inf
    n_epoch = 25

    for epoch in range(1, n_epoch + 1):
        train_loss = 0.0
        valid_loss = 0.0

        # Model training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            print('Batch %d' %(batch_idx))
            # Initialize weights to zero
            optimizer.zero_grad()

            output = model(data)

            # calculate loss
            loss = criterion(output, target)

            # Back propagation
            loss.backward()

            # gradient
            optimizer.step()


            train_loss = train_loss + loss.item()

            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d, loss  => %.6f' %(epoch, batch_idx + 1, train_loss))


    print('Finished training')
