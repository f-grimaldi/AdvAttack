import utils
import cifar10
import inception3

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch import nn, optim

if __name__ == '__main__':
    ### PARAMS
    BATCH_SIZE = 8
    NUM_WORKER = 2
    PRE_TRAINED    = True
    FREEZE_WEIGTHS = False

    EPOCH = 4
    LR    = 0.0002

    device = torch.device('cuda')
    ### DATA
    DataLoader = cifar10.CIFAR10()
    train, test = DataLoader.get_dataloader(BATCH_SIZE, NUM_WORKER)


    ### MODEL
    model = inception3.InceptionV3(pre_trained=PRE_TRAINED, freeze=FREEZE_WEIGTHS)
    model.net.fc = nn.Sequential(
                                 nn.Linear(in_features=2048, out_features=512, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5, inplace=False),
                                 nn.Linear(in_features=512, out_features=10, bias=True)
                                 )
    model.net = model.net.to(device)
    print(model.net)

    ### OPTIM, LOSS
    optimizer = optim.Adam(model.net.parameters(), lr=LR)
    loss_fn = loss_fn = nn.CrossEntropyLoss()

    ### TRAIN
    train_loss, test_loss = [], []
    accuracy_train, accuracy_test = [], []

    for ep in range(EPOCH):
        print('Epoch {}'.format(ep+1))
        time.sleep(0.3)
        ### TRAINING
        loss, accuracy = model.train_step(train, loss_fn, optimizer)
        train_loss.append(loss)
        accuracy_train.append(accuracy)

        ### EVALUATION
        loss, accuracy = model.eval_step(test, loss_fn)
        test_loss.append(loss)
        accuracy_test.append(accuracy)

        ### DISPLAY
        print('Training Loss: {}'.format(np.round(train_loss[-1], 4)), end = '\t')
        print('Training Accuracy Score: {}'.format(np.round(accuracy_train[-1], 4)))
        print('Test Loss:     {}'.format(np.round(test_loss[-1], 4)), end='\t')
        print('Test Accuracy Score:     {}\n'.format(np.round(accuracy_test[-1], 4)))
        time.sleep(0.3)
