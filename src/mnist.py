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
#import utils TO_FIX

"""
Take 1 channel Tensor and return a 3 channel Tensor with same values for each channel
"""
class RGB(object):

    def __call__(self, tensor):
        return tensor.expand((3, tensor.shape[1], tensor.shape[2]))

class MNIST(Dataset):
    """MNIST dataset."""

    def __init__(self, upscaling_dim=(299, 299), transform='standard', root='data/mnist'):
        """
        Args:
            dim       (tuple)                    The dimension of the upscaling
            transform (torchvision.transform)    The transfrom to apply. Default is 'standard' (ToTensor, Normalize, UpScaling)
            root      (string)                   The path of the root of the data
        """
        self.dim = upscaling_dim
        self.root = root
        self.transform = self.get_transform(transform)
        self.get_dataset()

    """
    Get the Transform
    """
    def get_transform(self, transform):
        if transform == 'standard':
            return transforms.Compose([transforms.Resize(self.dim, PIL.Image.LANCZOS),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)),
                                       RGB()])
        return transform

    """
    Get dataset
    """
    def get_dataset(self):
        self.train_dataset = datasets.MNIST(root=self.root, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=self.root, train=False, download=True, transform=self.transform)

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

        X, y = example[0][:5], example[1][:5]

        fig, ax = plt.subplots(1, 5, figsize=(20, 6))
        for n in range(5):
            img = np.transpose(X[n].numpy(), (1, 2, 0))
            if img.shape[2] == 1:
                img = img.reshape(img.shape[0]//2, -1)
            ax[n].imshow(img*0.3081 + 0.1307)
            ax[n].set_title(str(y[n].item()))
        plt.show()


if __name__ == '__main__':
    DataLoader = MNIST()
    train, test = DataLoader.get_dataloader(32, 4)
    DataLoader.get_info()
