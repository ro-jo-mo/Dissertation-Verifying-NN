import os
import torch
from torchvision import transforms
import torchvision
import arguments
from pathlib import Path

PATH = Path("dataset") / "GTSRB" / "Training"

def gtsrb_32_colour(spec):
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalise = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalise
    ])
    if spec["epsilon"] is None:
        raise ValueError('You must specify an epsilon')
    dataset = torchvision.datasets.ImageFolder(root=PATH,transform=transform)
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

def simple_cifar10(spec):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True,\
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data,\
            batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    if eps is None:
        raise ValueError('You must specify an epsilon')
    # Rescale epsilon.
    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps
