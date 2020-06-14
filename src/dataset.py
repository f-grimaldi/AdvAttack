# ### BASE
# from tqdm import tqdm
# ### CLASSIC
# import cv2


# ### TORCH
# import torch
# from torch.utils.data import Dataset, DataLoader

# from torch import nn, optim

# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
import torch
import PIL


class ToRGB(object):
    """
    RGB transformer
    Takes 1 channel Tensor and return a 3 channel Tensor with same values
    for each channel
    """

    def __call__(self, tensor):
        return tensor.expand((3, tensor.shape[1], tensor.shape[2]))


class MNIST(torch.utils.data.Dataset):
    """MNIST dataset"""

    def __init__(self, root='data/mnist', transform='standard', upscaling_dim=(299, 299)):
        """
        Args
        root (str):                         The root path of the dataset
        transform (torchvision.transforms): Transformation pipeline to apply
        upscaling_dim (tuple):              The dimension of the upscaling
        """
        self.dim = upscaling_dim
        self.root = root
        self.transform = self.get_transform(transform)
        self.fetch_dataset()

    def get_transform(self, transform):
        """
        Get the chosen transformer
        """
        if transform == 'standard':
            return transforms.Compose([
                transforms.ToTensor()
            ])

        elif transform == 'upscale':
            return transforms.Compose([
                transforms.Resize(self.dim, PIL.Image.LANCZOS),
                transforms.ToTensor(),
                ToRGB()
            ])
        elif type(transform) == torch.transforms.Compose:
            return transform
        else:
            raise NotImplementedError('Chosen transformer is not valid')

    def fetch_dataset(self):
        """
        Download MNIST dataset
        """
        self.train_dataset = datasets.MNIST(
            root=self.root,  # Define where dataset must be stored
            train=True,  # Retrieve training partition
            download=True,  # Retrieve dataset from remote repo
            transform=self.transform  # Apply chosen transforms
        )
        self.test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size, num_workers):
        """
        Get DataLoader instance from Dataset instance
        """
        # Get test dataset DataLoader object
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # Get train dataset DataLOader object
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        # Set training data loader
        return train_loader, test_loader

    @classmethod
    def get_info(cls, data_loader):
        """
        Given a DataLoader object, return some info
        """
        # Get first batch in given DataLoader object
        batch = next(iter(data_loader))
        # Print information
        print('\nBatch is a {}'.format(type(batch)))
        print('The 1st element is a {} with shape {}'.format(type(batch[0]), batch[0].shape))
        print('The 2nd element is a {} with shape {}'.format(type(batch[1]), batch[1].shape))
        # Get first 5 rows of X covariates and y target value
        X, y = batch[0][:5], batch[1][:5]
        # Make plot
        fig, ax = plt.subplots(1, 5, figsize=(20, 6))
        for n in range(5):
            img = np.transpose(X[n].detach().numpy(), (1, 2, 0))
            if img.shape[2] == 1:
                img = img.reshape(img.shape[0], img.shape[1])
            ax[n].imshow(img, cmap='gray')
            ax[n].set_title(str(y[n].item()))
        plt.show()


class CIFAR10(torch.utils.data.Dataset):
    """CIFAR10 dataset"""

    # Static attributes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, root='data/cifar10', transform='standard', upscaling_dim=(299, 299)):
        """
        Args
        root (str):                         The root path of the dataset
        transform (torchvision.transforms): Transformation pipeline to apply
        upscaling_dim (tuple):              The dimension of the upscaling
        """
        self.dim = upscaling_dim
        self.transform = self.get_transform(transform)
        self.root = root
        self.fetch_dataset()

    def get_transform(self, transform):
        if transform == 'standard':
            return transforms.Compose([
                transforms.ToTensor()
            ])
        if transform == 'upscale':
            return transforms.Compose([
                transforms.Resize(self.dim, PIL.Image.LANCZOS),
                transforms.ToTensor()
            ])
        elif type(transform) == torch.transforms.Compose:
            return transform
        else:
            raise NotImplementedError('Chosen transformer is not valid')

    def fetch_dataset(self):
        """
        Download CIFAR10 dataset
        """
        self.train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size, num_workers):
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return test_loader, train_loader

    @classmethod
    def get_info(cls, data_loader):
        # Retrieve first batch
        batch = next(iter(data_loader))
        # Print information
        print('\nBatch is a {}'.format(type(batch)))
        print('The 1st element is a {} with shape {}'.format(type(batch[0]), batch[0].shape))
        print('The 2nd element is a {} with shape {}'.format(type(batch[1]), batch[1].shape))
        # Get first 5 rows of X covariates and y target value
        X, y = batch[0][:5], batch[1][:5]
        # Make plot
        fig, ax = plt.subplots(1, 5, figsize=(20, 6))
        for n in range(5):
            img = np.transpose(X[n].numpy(), (1, 2, 0))
            ax[n].imshow(img)
            ax[n].set_title(cls.classes[y[n]])
        plt.show()


# Unit testing
if __name__ == '__main__':

    # Generate new CIFAR10 instance
    cifar10 = CIFAR10()
    # Get train and test data
    train, test = cifar10.get_dataloader(32, 4)
    # Get info for train data
    CIFAR10.get_info(train)
