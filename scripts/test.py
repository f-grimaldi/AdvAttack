def load_data(dataset, root, transform, batch_size, num_workers):
    if dataset == 'mnist':
        DataLoader = data.MNIST(root, transform=transform)
        train, test = DataLoader.get_dataloader(batch_size=batch_size, num_workers=num_workers)
    elif dataset == 'cifar10':
        DataLoader = data.CIFAR10(root, transform=transform)
        train, test = DataLoader.get_dataloader(batch_size=batch_size, num_workers=num_workers)
    else:
        print('Data not understood. Please use --data "mnist" or --data "cifar10"')
        raise NotImplementedError
        sys.exit(1)
    return DataLoader, train, test

def get_image(n, batch_size, device):
    if n >= 10000:
        print('The validation example are 10000. Please select an image index under 10000...')
        raise NotImplementedError
        sys.exit(1)
    n_batch = n//batch_size
    n_example = n%batch_size
    for i, batch in enumerate(test):
        if i == n_batch:
            X, y = batch[0][n_example].to(device), batch[1][n_example]
            break
        else:
            continue
    return X, y


def get_model(data):
    if data == 'mnist':
        model = models.MnistNet()
        model.load_state_dict(torch.load('../models/mnistBaseV2_state_dict.pth'))
    elif data == 'cifar10':
        model = models.MyVGG()
        model.load_state_dict(torch.load('../models/VGG16_cifar10_state_dict.pth'))
    return model

def check_output(X, y, model, device, is_softmax, dim):
    out = model(X.view(1, *list(X.shape)))
    y_pred = torch.argmax(out.cpu())
    if y_pred == y:
        if not is_softmax:
            return y, nn.Softmax(dim=dim)(out)[0, y]
        return y, out[0, y]
    print('Warnings: the model already misclassify current input. Setting the true target to the predicted one')
    if not is_softmax:
        return y_pred, nn.Softmax(dim=dim)(out)[0, y_pred]
    return y_pred, out[0, y_pred]

def get_loss(loss, target_neuron, maximise, is_softmax, softmax_dim):
    if loss == 'MSE':
        return customLoss.MSELoss(target_neuron, maximise, is_softmax, softmax_dim)
    elif loss == 'Zoo':
        transf = 0
        return customLoss.ZooLoss(target_neuron, maximise, transf, is_softmax, softmax_dim)
    else:
        print('{} is not supported. Please use either "Zoo" or "MSE"'.format(loss))
        raise NotImplementedError
        sys.exit(1)

def get_optimizer(optim, model, loss, device):
    if optim == 'inexact':
        return zeroOptim.InexactZSCG(model, loss, device)
    if optim == 'classic':
        return zeroOptim.ClassicZSCG(model, loss, device)
    if optim == 'zero_sgd':
        return zeroOptim.ZeroSGD(model, loss, device)
    else:
        print('Select one of the following optimizer:')
        print('Command                 Class           Descr')
        print('--optimizer "inexact"   InexactZSCG     Zero-order Stochastic Conditional Gradient with Inexact Updates')
        print('--optimizer "classic"   ClassicZSCG     Zero-order Stochastic Conditional Gradient')
        print('--optimizer "zero_sgd"  ZeroSGD         Zero-order Stochastic Gradient Descent')
        raise NotImplementedError
        sys.exit(1)

def get_optimization_params(optim, x, args):
    EPOCH = args.epochs
    if type(optim) == zeroOptim.InexactZSCG:
        params = {'x':x ,
                 'v':args.v, 'mk': [args.m]*EPOCH,
                 'mu_k':[args.mu]*EPOCH , 'gamma_k':[args.gamma]*EPOCH,
                 'C':args.C , 'epsilon':args.epsilon,
                 'max_steps':EPOCH, 'max_t': args.max_t,
                 'tqdm_disabled':args.tqdm_disabled, 'verbose': args.verbose}
    elif type(optim) == zeroOptim.ClassicZSCG:
        params = {'x':x ,
                 'v':args.v, 'mk': [args.m]*EPOCH,
                 'ak': [args.alpha]*EPOCH,
                 'C':args.C , 'epsilon':args.epsilon,
                 'max_steps':EPOCH,
                 'tqdm_disabled':args.tqdm_disabled, 'verbose': args.verbose}
    elif type(optim) == zeroOptim.ZeroSGD:
        params = {'x':x ,
                 'v':args.v, 'mk': [args.m]*EPOCH,
                 'ak': [args.lr]*EPOCH,
                 'C':args.C , 'epsilon':args.epsilon,
                 'max_steps':EPOCH,
                 'tqdm_disabled':args.tqdm_disabled, 'verbose': args.verbose}
    else:
        print('Select one of the following optimizer:')
        print('Command                 Class           Descr')
        print('--optimizer "inexact"   InexactZSCG     Zero-order Stochastic Conditional Gradient with Inexact Updates')
        print('--optimizer "classic"   ClassicZSCG     Zero-order Stochastic Conditional Gradient')
        print('--optimizer "zero_sgd"  ZeroSGD         Zero-order Stochastic Gradient Descent')
        raise NotImplementedError
        sys.exit(1)

    return params



if __name__ == '__main__':

    # 0. Set sys and import
    import sys
    import os
    import time
    import argparse
    import warnings
    from tqdm import tqdm

    cwd = os.getcwd()
    parent_cwd = cwd.split(os.sep)
    parent_cwd[-1] = 'src'
    src_path = os.sep.join(parent_cwd)
    sys.path.append(src_path)

    import torch
    import models
    from torch import nn, optim
    from torchvision import transforms


    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    import loss as customLoss
    import zeroOptim as zeroOptim
    import dataset as data



    # 1. Set argument
    parser = argparse.ArgumentParser(description='Zero-order attacks on black-box DNN')
    # 1.a) General args
    parser.add_argument('--seed',
                        type=int,               default=42,
                        help='Torch seed. Default is 42')
    parser.add_argument('--no_cuda',
                        action='store_true',    default=False,
                        help='disables CUDA training')
    # 1.b) Data ArgumentParser
    parser.add_argument('--data',
                        type=str,               default='mnist',
                        help='Dataset used to attack. Default mnist')
    parser.add_argument('--root',
                        type=str,               default='../data/',
                        help='Path of the dataset. Default data/')
    parser.add_argument('--transform',
                        type=str,               default='standard',
                        help='Default is standard. More detail on src/dataset.py')
    parser.add_argument('--batch_size',
                        type=int,               default=64,
                        help='Batch size loading the data. Default 64')
    parser.add_argument('--num_workers',
                        type=int,               default=0,
                        help='Number of workers for the dataloader. Default 0')
    parser.add_argument('--image_number',
                        type=int,               default=0,
                        help='Number of the image to use as variable')
    # 1.c) Model args
    parser.add_argument('--model_path',
                        type=str,               default='none',
                        help='Path of the model to attack')
    # 1.d) Loss args
    parser.add_argument('--loss',
                        type=str,               default='Zoo',
                        help='The type of loss. Either "Zoo" or "MSE". Default is "Zoo"')
    parser.add_argument('--target_neuron',
                        type=int,               default=-1,
                        help='If given a positive integer a specific/target attack is set to maximise the outcome of that neuron. Default is -1')
    parser.add_argument('--is_softmax',
                        action='store_true',    default=False,
                        help='Tell the loss that the model output is a propbability distribution already')
    parser.add_argument('--softmax_dim',
                        type=int,               default=1,
                        help='The dimension along which the softmax is applied')
    # 1.e) Optim general args
    parser.add_argument('--optimizer',
                        type=str,               default='inexact',
                        help='Supported optimizer: inexact, classic, zero_sgd. Default is inexact')
    parser.add_argument('--C',
                        type=tuple,         default=(0, 1),
                        help='Iterable with the minimum and the maximum value of a pixel. Default is (0, 1)')
    parser.add_argument('--epsilon',
                        type=float,             default=0.5,
                        help='The upper bound of L_infinity or L2 norm')
    parser.add_argument('--verbose',
                        type=int,             default=0,
                        help='The level of verbose. 0 no verbose, 1 partial, 2 total. Default is 0')
    parser.add_argument('--epochs',
                        type=int,             default=100,
                        help='The maximum number of steps')
    parser.add_argument('--max_time',
                        type=float,            default=1000,
                        help='The maximum number of seconds allowed to perform the attack')
    parser.add_argument('--tqdm_disabled',
                        type=int,              default=0,
                        help='If 1 the tqdm bar during the optimization procedure will not be displayed. Default is 0')
    # 1.d) General arguments in Balasubramanian Algortihms
    parser.add_argument('--v',
                        type=float,            default=0.001,
                        help='The gaussian smoothing parameters. Usually the lesser the more precise is the gradient. Default is 0.001')
    parser.add_argument('--m',
                        type=int,              default=1000,
                        help='The number of gaussian vector generated  for computing the pseudo gradient. Warning: first cause of OutOfMemory. Default 1000')
    # 1.e) Classic ZSCG args
    parser.add_argument('--alpha',
                        type=float,            default=0.2,
                        help='The momentum/lr of Classic ZSCG. Must be in (0, 1). Suggested is 1/sqrt(EPOCH). Default is 0.2. Only for Classic ZSCG')
    # 1.f) Inexact ZSGG args
    parser.add_argument('--gamma',
                        type=float,            default=2,
                        help='Gradient update rule inside ICG. Suggested is 2L. Only for Inexact ZSCG')
    parser.add_argument('--mu',
                        type=float,            default=1/400,
                        help='The first stopping criterion inside ICG. Only for Inexact ZSCG')
    parser.add_argument('--max_t',
                        type=int,              default=50,
                        help='The maximum number of steps inside ICG. Default is 50. Only for Inexact ZSCG')
    # 1.g) Zero SGD args
    parser.add_argument('--lr',
                        type=float,             default=0.1,
                        help='The learning rate for the Zero SGD. Default is 0.01. Only for Zero SGD')

    args = parser.parse_args()
    print('Arguments are:\n{}'.format(args))

    # 2. Set device and sees
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device is: {}'.format(device))

    # 3. Load data
    root = '{}{}'.format(args.root, args.data)
    dataloader, train, test = load_data(args.data, root, args.transform, args.batch_size, args.num_workers)

    # 4. Loading the network
    if args.model_path != 'none':
        try:
            net = torch.load(args.model_path).to(device)
        except:
            print('Couldnt load the model from path {}. Either wrong path or model class not found in src/models.')
            print('Using default model for given dataset')
            model = get_model(args.data).to(device)
    else:
        model = get_model(args.data).to(device)

    # 4. Chose image to use as variable
    X, y = get_image(args.image_number, args.batch_size, device)

    y, original_out = check_output(X, y, model, device, args.is_softmax, args.softmax_dim)

    # 6. Set the loss function
    if args.target_neuron == -1:
        maximise = 0
        loss_fn = get_loss(args.loss, int(y), maximise, args.is_softmax, args.softmax_dim)
    elif args.target_neuron > -1 and args.target_neuron < 10:
        maximise = 1
        loss_fn = get_loss(args.loss, args.target_neuron , maximise, args.is_softmax, args.softmax_dim)
    else:
        print('Please select a target neuron in [0, 9]')
        print(sys.exit(1))

    # 7. Set optimizer and run parameters
    optim = get_optimizer(args.optimizer, model, loss_fn, device)
    params = get_optimization_params(optim, X, args)

    # 8. Perform the run
    new_x, loss_list, out_list = optim.run(**params)

    # 9. Display results
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # 9.a) Loss
    ax[0].plot(loss_list)
    ax[0].set_title('Loss curve')
    ax[0].set_ylabel('Loss {}'.format(args.loss))
    ax[0].set_xlabel('Step')
    ax[0].grid()

    # 9.b) Image
    new_out = model(new_x.view(1, *list(new_x.shape)))
    if not args.is_softmax:
        softmax = nn.Softmax(dim=args.softmax_dim)
        new_out = softmax(new_out)
    new_y = torch.argmax(new_out)

    if new_x.shape[0] > 1:
        ax[1].imshow(np.transpose(new_x.cpu().numpy(), 1, 2, 0))
    else:
        ax[1].imshow(new_x.cpu().numpy().reshape(new_x.shape[1], new_x.shape[2]))
    ax[1].set_title('After attack\nP(X={}) = {:.3f}\nP(X={}) = {:.3f}'.format(y, new_out[0, y], new_y, new_out[0, new_y]))
    plt.show()
