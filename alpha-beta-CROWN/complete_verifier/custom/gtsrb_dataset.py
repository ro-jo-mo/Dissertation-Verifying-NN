import os
import torch
from torchvision import transforms
import torchvision
import arguments
from pathlib import Path

PATH = Path("/home/rob/code/Project/dataset/GTSRB/Training")
torch.manual_seed(42)
def gtsrb_32_colour(spec):
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalise = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalise
    ])
    epsilon = spec["epsilon"]
    if epsilon is None:
        raise ValueError("You must specify an epsilon")
    # Load full dataset 
    dataset = torchvision.datasets.ImageFolder(root=PATH, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=99999, 
                                           num_workers=6,
                                           shuffle=False
                                           )
    images,labels = next(iter(loader))
    minimum_values = normalise(torch.zeros((1,3,1,1)))
    maximum_values = normalise(torch.ones((1,3,1,1)))
    epsilon = (epsilon / std).reshape(1,3,1,1)
    return images, labels, maximum_values, minimum_values, epsilon

def get_n_of_each(imgs,labels,n): 
    n_classes = 43
    _N = 32
    frequency = torch.zeros(n_classes)
    save_imgs = torch.zeros((n*n_classes,1,_N,_N),dtype=torch.float32)
    save_labels = torch.zeros(n*n_classes,dtype=torch.int64)
    counter = 0
    for img,label in zip(imgs,labels):
        if frequency[label] >= n:
            continue
        frequency[label] += 1
        save_imgs[counter] = img
        save_labels[counter] = label
        counter += 1
    return save_imgs,save_labels


def gtsrb_32_grey(spec):
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalise = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Grayscale(),
        normalise
    ])
    epsilon = spec["epsilon"]
    if epsilon is None:
        raise ValueError("You must specify an epsilon")
    # Load full dataset 
    dataset = torchvision.datasets.ImageFolder(root=PATH, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=99999, 
                                           num_workers=6,
                                           shuffle=True
                                           )
                                           
    images,labels = next(iter(loader))
    images,labels = get_n_of_each(images,labels,5)
    minimum_values = normalise(torch.zeros((1,1,1,1)))
    maximum_values = normalise(torch.ones((1,1,1,1)))
    epsilon = (epsilon / std).reshape(1,1,1,1)
    return images, labels, maximum_values, minimum_values, epsilon
