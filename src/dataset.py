# Dependencies
# Default dependencies
import matplotlib.pyplot as plt
import numpy as np
import PIL
# Torch dependencies
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class RGB(object):
    """
    Take 1 channel Tensor and return a 3 channel Tensor with same values
    for each channel
    """

    def __call__(self, tensor):
        return tensor.expand((3, tensor.shape[1], tensor.shape[2]))


class MNIST(Dataset):
    """
    MNIST dataset
    """

    def __init__(
        self, root='data/mnist', transform='default',
        upscaling_dim=(299, 299)
    ):
        """
        Args:
        root (str):                         The root path of the dataset;
        transform (torchvision.transforms): The transform pipeline to apply;
        upscaling_dim (tuple):              The dimension of the upscaling;
        """
        self.dim = upscaling_dim
        self.root = root
        self.transform = self.get_transform(transform)
        self.fetch_dataset()

    def get_transform(self, transform='default'):
        """
        Get the required transformation pipeline
        """
        if transform == 'default':  # Default: just parse to tensor
            return transforms.Compose([
                transforms.ToTensor()
            ])
        elif transform == 'upscale':  # Upscaling
            return transforms.Compose([
                transforms.Resize(self.dim, PIL.Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                RGB()
            ])
        elif type(transform) == transforms.Compose:  # Custom transform
            return transform
        else:  # No valid transform
            raise NotImplementedError(
                'Please provide a valid transformation pipeline'
            )

    def fetch_dataset(self):
        """
        Fetch MNIST dataset from remote repositories
        """
        # Fetch train dataset
        self.train_dataset = datasets.MNIST(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform
        )
        # Fetch test dataset
        self.test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size, num_workers):
        """
        Get train and test DataLoader objects
        """
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # Define test data loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # Define train data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        # Return train and test DataLoader objects
        return train_loader, test_loader

    @classmethod
    def get_info(cls, data_loader):
        """
        Shows some infor for a MNIST train or test DataLoader object
        """
        # Get first example batch
        batch = next(iter(data_loader))
        print('\nBatch is a {}'.format(type(batch)))
        print('The 1st element is a {0:s} with shape {1:s}'.format(
            str(type(batch[0])), str(batch[0].shape)
        ))
        print('The 2nd element is a {0:s} with shape {1:s}'.format(
            str(type(batch[1])), str(batch[1].shape)
        ))
        # Get some entries from batch
        X, y = batch[0][:5], batch[1][:5]
        # Make plot
        fig, ax = plt.subplots(1, 5, figsize=(20, 6))
        # Loop through the first 5 rows of the selected batch
        for i in range(5):
            # Transpose the image as a numpy matrix
            img = np.transpose(X[i].numpy(), (1, 2, 0))
            # CEventually reshape image
            if img.shape[2] == 1:
                img = img.reshape(img.shape[0]//2, -1)
            # Make image subplot
            _ = ax[i].imshow(img*0.3081 + 0.1307)
            _ = ax[i].set_title(str(y[i].item()))
        # Show plot
        _ = plt.show()


class CIFAR10(Dataset):
    """
    CIFAR10 dataset
    """

    # Static attributes
    classes = tuple([  # Target classes
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])

    # Constructor
    def __init__(self, root='data/cifar10', transform='default', upscaling_dim=(299, 299)):
        """
        Args:
        root (str):                         The root path of the dataset;
        transform (torchvision.transforms): The transform pipeline to apply;
        upscaling_dim (tuple):              The dimension of the upscaling;
        """
        self.dim = upscaling_dim
        self.transform = self.get_transform(transform)
        self.root = root
        self.fetch_dataset()

    def get_transform(self, transform='default'):
        """
        Get the transformation pipeline
        """
        if transform == 'default':  # Default
            return transforms.Compose([
                transforms.ToTensor()
            ])
        if transform == 'upscale':  # Upscaling
            return transforms.Compose([
                transforms.Resize(self.dim, PIL.Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif type(transform) == torch.transforms.Compose:  # Custom transform
            return transform
        else:  # No valid transform
            raise NotImplementedError(
                'Please provide a valid transformation pipeline'
            )

    def fetch_dataset(self):
        """
        Fetch MNIST dataset from remote repositories
        """
        # Fetch train dataset
        self.train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform
        )
        # Fetch test dataset
        self.test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size, num_workers):
        """
        Get train and test DataLoader objects
        """
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # Define test data loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # Define train data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        # Return train and test DataLoader objects
        return train_loader, test_loader

    @classmethod
    def get_info(cls, data_loader):
        """
        Shows some infor for a CIFAR10 train or test DataLoader object
        """
        # Get first example batch
        batch = next(iter(data_loader))
        print('\nBatch is a {}'.format(type(batch)))
        print('The 1st element is a {0:s} with shape {1:s}'.format(
            str(type(batch[0])), str(batch[0].shape)
        ))
        print('The 2nd element is a {0:s} with shape {1:s}'.format(
            str(type(batch[1])), str(batch[1].shape)
        ))
        # Get some entries from batch
        X, y = batch[0][:5], cls.classes[1][:5]
        # Make plot
        fig, ax = plt.subplots(1, 5, figsize=(20, 6))
        # Loop through the first 5 rows of the selected batch
        for i in range(5):
            # Transpose the image as a numpy matrix
            img = np.transpose(X[i].numpy(), (1, 2, 0))
            # Make image subplot
            _ = ax[i].imshow(img/2 + 0.5)
            _ = ax[i].set_title(str(y[i]))
        # Show plot
        _ = plt.show()


# Test
if __name__ == '__main__':
    # Instantiate new CIFAR10 dataset
    cifar10 = CIFAR10()
    # Split in train and test dataset
    train, test = cifar10.get_dataloader(32, 4)
    # Get train dataset info
    CIFAR10.get_info(train)
    # Get test dataset info
    CIFAR10.get_info(test)
