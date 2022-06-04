import os
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess_test = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])
preprocess_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(90, interpolation=False, expand=False, center=None),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])


class DatasetHagni40(Dataset):
    def __init__(self, root, tag, mode) -> None:
        super().__init__()
        self.dir = os.path.join(root, tag)
        self.mode = mode
        f = open(self.dir, 'r')
        list_mes = f.readlines()
        self.list_mes = [ele.strip('\n') for ele in list_mes]
        self.data_size = len(self.list_mes)

    def __getitem__(self, index):
        element = self.list_mes[index]

        img_path = element.split(' ')[0] + ' ' + element.split(' ')[1]
        image = Image.open(img_path)
        type = element.split(' ')[2]
        target = int(element.split(' ')[3].strip(' '))
        if self.mode:
            image = preprocess_train(image)
        else:
            image = preprocess_test(image)
        target = torch.tensor(target)

        return image, target, type

    def __len__(self):
        return self.data_size


# define train loader
def train_test_dataloader(data_dir, batch_size, num_workers, mode):
    full_dataset = DatasetHagni40(data_dir, 'Hagni40.txt', mode)
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                                          [train_size, test_size, val_size],
                                                                          generator=torch.Generator().manual_seed(40))
    # 40 is the random number seed, which can be set freely

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader, test_loader, val_loader