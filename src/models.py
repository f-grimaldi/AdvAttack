import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torchvision import models
from torch import nn, optim

class MnistNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.FeatureExtractor = nn.Sequential(nn.Conv2d(1, 8, (2, 2), stride=1, padding=1), nn.ReLU(),
                                              nn.Conv2d(8, 16, (2, 2), stride=1, padding=1), nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=(2, 2)),
                                              nn.Conv2d(16, 20, (2, 2), stride=1, padding=1), nn.ReLU(),
                                              nn.Conv2d(20, 24, (2, 2), stride=1, padding=1), nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=(2, 2)))

        self.Classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(24*8*8, 256), nn.Dropout(0.3), nn.ReLU(),
                                        nn.Linear(256, 64), nn.Dropout(0.3), nn.ReLU(),
                                        nn.Linear(64, 10))


    def forward(self, x):
        cnn_x = self.FeatureExtractor(x)
        out = self.Classifier(cnn_x)
        return out

    def train_step(self, trainloader, loss_fn, optimizer, device):
         ### 1. Train
        self.train()
        ### 1.1 Define vars
        loss, accuracy = [], []

        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            ### 1.2 Feed the network
            X, y = batch[0].to(device), batch[1].to(device)
            out = self(X)
            ### 1.3 Compute loss and back-propagate
            crn_loss = loss_fn(out, y)
            crn_loss.backward()
            optimizer.step()
            ### 1.4 Save results
            loss.append(crn_loss.data.item())
            accuracy.append(accuracy_score(batch[1].numpy(), np.argmax(out.cpu().detach().numpy(), axis=1)))

        return np.mean(loss), np.mean(accuracy)

    def eval_step(self, testloader, loss_fn, device):
        ### 1. Eval
        self.eval()
        ### 1.1 Define vars
        loss, accuracy = [], []

        with torch.no_grad():
            for batch in tqdm(testloader):
                ### 1.2 Feed the network
                X, y = batch[0].to(device), batch[1].to(device)
                out = self(X)
                ### 1.3 Compute loss
                crn_loss = loss_fn(out, y)
                ### 1.4 Save results
                loss.append(crn_loss.data.item())
                accuracy.append(accuracy_score(batch[1], np.argmax(out.cpu().detach().numpy(), axis=1)))

        return np.mean(loss), np.mean(accuracy)


class InceptionV3():

    def __init__(self, pre_trained=True, freeze=False):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Found device: {}'.format(self.device))

        self.net = models.inception_v3(pretrained=pre_trained).to(self.device)
        if freeze:
            for param in self.net.parameters():
                param.require_grad = False

    def train_step(self, trainloader, loss_fn, optimizer):
         ### 1. Train
        self.net.train()
        ### 1.1 Define vars
        loss, accuracy = [], []

        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            ### 1.2 Feed the network
            X, y = batch[0].to(self.device), batch[1].to(self.device)
            out = self.net(X)[0]
            ### 1.3 Compute loss and back-propagate
            crn_loss = loss_fn(out, y)
            crn_loss.backward()
            optimizer.step()
            # print(batch[1].shape)
            # print(out.cpu().detach().numpy().shape)
            ### 1.4 Save results
            loss.append(crn_loss.data.item())
            accuracy.append(accuracy_score(batch[1].numpy(), np.argmax(out.cpu().detach().numpy(), axis=1)))

        return np.mean(loss), np.mean(accuracy)

    def eval_step(self, testloader, loss_fn):
        ### 1. Eval
        self.net.eval()
        ### 1.1 Define vars
        loss, accuracy = [], []

        with torch.no_grad():
            for batch in tqdm(testloader):
                ### 1.2 Feed the network
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                out = self.net(X)
                ### 1.3 Compute loss
                crn_loss = loss_fn(out, y)
                ### 1.4 Save results
                loss.append(crn_loss.data.item())
                accuracy.append(accuracy_score(batch[1], np.argmax(out.cpu().detach().numpy(), axis=1)))

        return np.mean(loss), np.mean(accuracy)


"""
Create a Network composed by 2 Sequential modules:
    1. A part of the VGG16 pre-trained CNN Sequential module. The part is defined by the variable "last_feature_block"
    2. A dense Sequential module of 4 layers with a final layer of 10 neurons in order to classify the CIFAR10 dataset.
       The two hidden layer size defined by "dense_dim".
"""
class MyVGG(nn.Module):
    """
    dense_dim               tuple         Define the the hidden layer sizes of the two hidden layer of the Dense
                                          Sequential Module. Default is (1024, 128)
    last_feature_block      int           Define the number of layers of the VGG CNN Sequential module we copy.
                                          Default 34 (All block).
    set_trainable           bool          If False the Convolutional Layers of vgg16 are frozen
                                          (requires_grad is set to False). Default True

    """
    def __init__(self, dense_dim = (1024, 128), last_feature_block = 34, set_trainable=True):
        ### 1. Init nn.Module
        super(MyVGG, self).__init__()

        ## 2. Define attributes
        self.last_block = last_feature_block
        self.set_trainable = set_trainable
        self.dense_dim = dense_dim

        ### 3. Create Network
        self.CNN = self.init_cnn(models.vgg16_bn(pretrained=True,
                                                 progress=False))                   # Create Convolutional Module
        self.cnn_dim = self.get_dim()                                               # Get dims of the 1st module out
        self.Flatten = nn.Flatten()                                                 # Flatten the input
        self.Dense = self.init_dense()                                              # Create Dense Module

    def get_dim(self):
        out = self.CNN(torch.rand([1, 3, 32, 32]))
        return out.shape[1:]

    def init_cnn(self, vgg):
        if self.set_trainable == False:
            for param in vgg.features.parameters():
                param.requires_grad = False
        return nn.Sequential(*[module for module in vgg.features[:self.last_block]])

    def init_dense(self):
        l1 = nn.Linear(in_features=torch.prod(torch.tensor(self.cnn_dim)), out_features=self.dense_dim[0], bias=True)
        r1 = nn.ReLU(inplace=True)
        d1 = nn.Dropout(p=0.5, inplace=False)
        l2 = nn.Linear(in_features=self.dense_dim[0], out_features=self.dense_dim[0], bias=True)
        r2 = nn.ReLU(inplace=True)
        d2 = nn.Dropout(p=0.5, inplace=False)
        l3 = nn.Linear(in_features=self.dense_dim[0], out_features=self.dense_dim[1], bias=True)
        r3 = nn.ReLU(inplace=True)
        d3 = nn.Dropout(p=0.5, inplace=False)
        l4 = nn.Linear(in_features=self.dense_dim[1], out_features=10, bias=True)
        return nn.Sequential(*[l1, r1, d1, l2, r2, d2, l3, r3, d3, l4])

    def forward(self, x):
        cnn = self.CNN(x)
        flatten = self.Flatten(cnn)
        out = self.Dense(flatten)
        return out
