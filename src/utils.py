### BASE
from tqdm import tqdm
### CLASSIC
import cv2
import numpy as np
### TORCH
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim


"""
Get batches
"""
def get_batch(trainset, testset, train_batch=32, test_batch=32):
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                             shuffle=True, num_workers=2)
    return trainloader, testloader

"""
Upscale images
"""
def upscale(batch, dim = (299, 299)):
    result = np.zeros((batch.shape[0], dim[0], dim[1], 3))
    numpy_input = np.transpose(batch.numpy(), (0, 2, 3, 1))
    for n, img in enumerate(numpy_input):
        result[n, ...] = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    result = np.transpose(result, (0, 3, 1, 2))
    out = torch.tensor(result).float()
    return out
