import utils
import cifar10

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torchvision import models
from torch import nn, optim

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
