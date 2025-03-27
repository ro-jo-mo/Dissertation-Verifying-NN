import torch
import torchvision
from torchvision import transforms
from pathlib import Path

# from dataset import get_data
# PATH,LABELS,normalise,GROUPS,GROUP_NAMES,n_classes,train_loader,test_loader = get_data(_N,batch_size)

def get_data(_N,batch_size):
    torch.manual_seed(42)
    mean = 0.3211
    std = 0.2230
    PATH = Path("/home/rob/code/Project/dataset/GTSRB/Training")
    LABELS = ["20 km/h"
    ,"30 km/h"
    ,"50 km/h"
    ,"60 km/h"
    ,"70 km/h"
    ,"80 km/h"
    ,"End of speed limit"
    ,"100 km/h"
    ,"120 km/h"
    ,"No passing"
    ,"No passing for large vehicles"
    ,"Right of way"
    ,"Priority road"
    ,"Yield"
    ,"Stop"
    ,"No vehicles"
    ,"Large vehicles prohibited"
    ,"No entry"
    ,"General caution"
    ,"Dangerous curve to the left"
    ,"Dangerous curve to the right"
    ,"Double curve"
    ,"Bumpy road"
    ,"Slippery road"
    ,"Road narrows"
    ,"Road work"
    ,"Traffic signals"
    ,"Pedestrians"
    ,"Children crossing"
    ,"Bicycle crossing"
    ,"Ice warning"
    ,"Wild animals crossing"
    ,"End of all speed and passing limits"
    ,"Turn right ahead"
    ,"Turn left ahead"
    ,"Ahead only"
    ,"Go straight or right"
    ,"Go straight or left"
    ,"Keep right"
    ,"Keep left"
    ,"Roundabout mandatory"
    ,"End of no passing"
    ,"End of no passing by large vehicles"]
    


    normalise = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        transforms.Resize((_N,_N)),
        transforms.ToTensor(),
        transforms.Grayscale(),
        normalise
    ])
    _dataset = torchvision.datasets.ImageFolder(root=PATH, transform=transform)
    loader = torch.utils.data.DataLoader(_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=6,
                                        shuffle=True
                                        )

    def get_mean_std():
        transform = transforms.Compose([
        transforms.Resize((_N,_N)),
        transforms.ToTensor(),
        transforms.Grayscale()
        ])
        dataset = torchvision.datasets.ImageFolder(root=PATH,transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mean = 0
        std = 0
        total = 0
        for images,_ in dataloader: # load a batch at a time
            total += 1
            mean  += images.mean()
            std   += images.std()
        std  /= total
        mean /= total
        return mean,std
    # get_mean_std() # (tensor(0.3211), tensor(0.2230))

    GROUPS = [3,3,3,3,3,3,2,3,3,4,4,1,0,0,0,4,4,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,5,5,5,5,5,5,5,5,2,2] 
    GROUP_NAMES = ["Unique Signs","Danger Signs","Derestriction Signs","Speed Limit Signs","Other Prohibitory Signs","Mandatory Signs"]
    groups_to_keep = {0,2,4}
    def filter_dataset(dataset):
        labels_to_keep = {i for i in range(43) if GROUPS[i] in groups_to_keep}
        to_keep = [i for i,(_,label) in enumerate(dataset) if label in labels_to_keep]
        new_label_mapping = {old : new for new,old in enumerate(labels_to_keep)}
        remap = lambda x : new_label_mapping[x]
        dataset.target_transform = remap
        filtered_dataset = torch.utils.data.Subset(dataset,to_keep)
        return filtered_dataset, new_label_mapping
    filtered_dataset, mapping = filter_dataset(_dataset)
    n_classes = len(mapping)
    LABELS = [LABELS[old] for old in mapping]
    def update_groups(GROUPS,GROUP_NAMES):
        GROUP_NAMES = [GROUP_NAMES[i] for i in groups_to_keep]
        remap = {old : new for new,old in enumerate(groups_to_keep)}
        groups = [remap[GROUPS[old]] for old in mapping]
        return groups, GROUP_NAMES
    GROUPS, GROUP_NAMES = update_groups(GROUPS,GROUP_NAMES)

    train_dataset, test_dataset = torch.utils.data.random_split(filtered_dataset, [0.8,0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            num_workers=6,
                                            shuffle=True
                                            )
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            num_workers=6,
                                            shuffle=False
                                            )
    return PATH,LABELS,normalise,GROUPS,GROUP_NAMES,n_classes,train_loader,test_loader