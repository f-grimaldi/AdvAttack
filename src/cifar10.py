### BASE
from tqdm import tqdm
### CLASSIC
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
### TORCH
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch import nn, optim
### CUSTOM
import utils

class CIFAR10(Dataset):
    """CIFAR10 dataset."""

    def __init__(self, upscaling_dim=(299, 299), transform='standard'):
        """
        Args:
            dim  (tuple):                        The dimension of the upscaling
            transform (torchvision.transform):   The transfrom to apply. Default is 'standard' (ToTensor, Normalize, UpScaling)
        """
        self.dim = upscaling_dim
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform = self.get_transform(transform)
        self.get_dataset()

    """
    Get the Transform
    """
    def get_transform(self, transform):
        if transform == 'standard':
            return transforms.Compose([transforms.Resize(self.dim, PIL.Image.LANCZOS),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform

    """
    Get dataset
    """
    def get_dataset(self):
        self.train_dataset = datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=self.transform)

    """
    Get dataloader
    """
    def get_dataloader(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=num_workers)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=num_workers)
        return self.trainloader, self.testloader


    """
    Visualize first n example
    """
    def get_info(self):
        for batch in iter(self.trainloader):
            example = batch
            print('\nBatch is a {}'.format(type(batch)))
            print('The 1st element is a {} with shape {}'.format(type(batch[0]), batch[0].shape))
            print('The 2nd element is a {} with shape {}'.format(type(batch[1]), batch[1].shape))
            break

        batch = example[0][:5]
        fig, ax = plt.subplots(1, 5, figsize=(20, 6))
        for n in range(5):
            img = np.transpose(batch[n].numpy(), (1, 2, 0))
            ax[n].imshow(img/2 + 0.5)
            ax[n].set_title(self.classes[example[1][n]])
        plt.show()


if __name__ == '__main__':
    DataLoader = CIFAR10()
    train, test = DataLoader.get_dataloader(32, 4)
    DataLoader.get_info()
