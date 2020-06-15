def load_data(dataset, is_inception, root, transform, batch_size, num_workers):

    if is_inception and transform == 'standard':
        print('Trying to use inceptionV3 without upscaling!')
        print('Use --transform "upscale" when --use_inception')
        sys.exit(1)
    if not is_inception and transform != 'standard':
        print('Trying to use base models but calling data upscaling!')
        print('Use --transform "standard" when not using "use_inception"')
        sys.exit(1)

    if is_inception and transform == 'upscale':
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


def get_model(is_inception, data, device):
    if not is_inception:
        if data == 'mnist':
            model = models.MnistNet()
            model.load_state_dict(torch.load('../models/mnistBaseV2_state_dict.pth'))
        elif data == 'cifar10':
            model = models.MyVGG()
            model.load_state_dict(torch.load('../models/VGG16_cifar10_state_dict.pth'))
    else:
        model = models.InceptionV3().net
        model.fc = nn.Sequential(nn.Linear(2048, 512), nn.Dropout(0.4), nn.ReLU(),
                                 nn.Linear(512, 64), nn.Dropout(0.4), nn.ReLU(),
                                 nn.Linear(64, 10)).to(device)
        if data == 'mnist':
            model.load_state_dict(torch.load('../models/InceptionV3_MNIST_state_dict.pth'))
        elif data == 'cifar10':
            model.load_state_dict(torch.load('../models/InceptionV3_CIFAR10_state_dict.pth'))

    return model


def check_output(X, y, model, device, is_softmax, dim):
    out = model(X.view(1, *list(X.shape)))
    y_pred = torch.argmax(out.cpu())
    if y_pred == y:
        if not is_softmax:
            return y, nn.Softmax(dim=dim)(out)
        return y, out
    print('Warnings: the model already misclassify current input. Setting the true target to the predicted one')
    if not is_softmax:
        return y_pred, nn.Softmax(dim=dim)(out)
    return y_pred, out


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
    elif optim == 'accelerated':
        return zeroOptim.InexactAcceleratedZSCG(model, loss, device)
    elif optim == 'classic':
        return zeroOptim.ClassicZSCG(model, loss, device)
    elif optim == 'zero_sgd':
        return zeroOptim.ZeroSGD(model, loss, device)
    elif optim == 'zoo':
        return ZOOptim.ZOOptim(model, loss, device)
    elif optim == 'fw':
        return FWOptim.FrankWolfe(model, loss, device)
    else:
        print('Select one of the following optimizer:')
        print('Command                     Class                    Descr')
        print('--optimizer "inexact"       InexactZSCG              Zero-order Stochastic Conditional Gradient with Inexact Updates')
        print('--optimizer "classic"       ClassicZSCG              Zero-order Stochastic Conditional Gradient')
        print('--optimizer "accelerated"   InexactAcceleratedZSCG   Accelerated Zero-order Stochastic Conditional Gradient with Inexact Updates')
        print('--optimizer "zero_sgd"      ZeroSGD                  Zero-order Stochastic Gradient Descent')
        print('--optimizer "zoo"           ZOOptim                  Zero-order Stochastic Gradient Descent with Coordinate-wise ADAM/Newton')
        print('--optimizer "fw"            FWOptim                  Zero-order Frank-Wolfe Gradient Descent')
        raise NotImplementedError
        sys.exit(1)


def get_optimization_params(optim, x, args):
    EPOCH = args.epochs
    if args.batch_size == -1:
        bs = args.n_gradient
    else:
        bs = args.batch_size
    if type(optim) == zeroOptim.InexactZSCG:
        params = {'x': x,
                  'v': args.v,
                  'n_gradient': [args.n_gradient]*EPOCH,
                  'batch_size': bs,
                  'mu_k': [args.mu]*EPOCH,
                  'gamma_k': [args.gamma]*EPOCH,
                  'C': args.C,
                  'epsilon': args.epsilon,
                  'L_type': args.L_type,
                  'max_steps': EPOCH,
                  'max_t': args.max_t,
                  'tqdm_disabled': args.tqdm_disabled,
                  'verbose': args.verbose}
    elif type(optim) == zeroOptim.InexactAcceleratedZSCG:
        params = {'x': x,
                  'v': args.v,
                  'n_gradient': [args.n_gradient]*EPOCH,
                  'batch_size': bs,
                  'alpha_k': [args.alpha]*EPOCH,
                  'mu_k': [args.mu]*EPOCH,
                  'gamma_k': [args.gamma]*EPOCH,
                  'C': args.C,
                  'epsilon': args.epsilon,
                  'L_type': args.L_type,
                  'max_steps': EPOCH,
                  'max_t': args.max_t,
                  'tqdm_disabled': args.tqdm_disabled,
                  'verbose': args.verbose}
    elif type(optim) == zeroOptim.ClassicZSCG:
        params = {'x': x,
                  'v': args.v,
                  'n_gradient': [args.n_gradient]*EPOCH,
                  'L_type': args.L_type,
                  'batch_size': bs,
                  'ak': [args.alpha]*EPOCH,
                  'C': args.C ,
                  'epsilon': args.epsilon,
                  'max_steps': EPOCH,
                  'tqdm_disabled': args.tqdm_disabled,
                  'verbose': args.verbose}
    elif type(optim) == zeroOptim.ZeroSGD:
        params = {'x': x,
                  'v': args.v,
                  'n_gradient': [args.n_gradient]*EPOCH,
                  'batch_size': bs,
                  'L_type': args.L_type,
                  'ak': [args.lr]*EPOCH,
                  'C': args.C,
                  'epsilon': args.epsilon,
                  'max_steps': EPOCH,
                  'tqdm_disabled': args.tqdm_disabled,
                  'verbose': args.verbose}
    elif type(optim) == ZOOptim.ZOOptim:
        params = {'x': x,
                  'c': args.c,
                  'n_gradient': args.n_gradient,
                  'batch_size': bs,
                  'beta_1': args.beta_1,
                  'beta_2': args.beta_2,
                  'solver': args.solver,
                  'stop_criterion': args.stop_criterion,
                  'learning_rate': args.lr,
                  'max_steps': EPOCH,
                  'C': args.C,
                  'tqdm_disabled': args.tqdm_disabled,
                  'verbose': args.verbose}

    elif type(optim) == FWOptim.FrankWolfe:
        if args.L_type == -1:
            l_type = 'inf'
        if args.verbose == 0:
            verbose = False
        else:
            verbose = True
        params = {
            'x': x,
            'grad_num_iter': args.n_gradient,
            'grad_batch_size': bs, #args.batch_size
            'm_weight': args.FW_beta,
            'grad_smooth': args.FW_delta,
            'step_size': args.FW_gamma,
            'l_bound': args.epsilon,
            'l_type': l_type, #args.L_type
            'num_epochs': EPOCH, #args.epochs
            'clip': args.C,
            'verbose': verbose
        }

    else:
        print('Select one of the following optimizer:')
        print('Command                     Class                    Descr')
        print('--optimizer "inexact"       InexactZSCG              Zero-order Stochastic Conditional Gradient with Inexact Updates')
        print('--optimizer "classic"       ClassicZSCG              Zero-order Stochastic Conditional Gradient')
        print('--optimizer "accelerated"   InexactAcceleratedZSCG   Accelerated Zero-order Stochastic Conditional Gradient with Inexact Updates')
        print('--optimizer "zero_sgd"      ZeroSGD                  Zero-order Stochastic Gradient Descent')
        print('--optimizer "zoo"           ZOOptim                  Zero-order Stochastic Gradient Descent with Coordinate-wise ADAM/Newton')
        print('--optimizer "fw"            FWOptim                  Zero-order Frank-Wolfe Gradient Descent')
        raise NotImplementedError
        sys.exit(1)

    return params


def get_data(n_example, data_batch_size, dataloader, device):
    if n_example < data_batch_size:
        raise ValueError('n_exaple is less than data_batch_size. Please use --data_batch_size n --n_example m with m >= n')
    n_batch = n_example//data_batch_size
    X_ori = torch.Tensor()
    y_ori = torch.Tensor().long()
    for n, batch in enumerate(dataloader):
        if n == n_batch:
            break
        y_ori = torch.cat((y_ori, batch[1]))
        X_ori = torch.cat((X_ori, batch[0]))
    return X_ori, y_ori


def untarget_run(args, model, X_ori, y_ori, device):
    success = []
    loss_list = []
    epsilon = []
    time_t = []
    for X, y in tqdm(zip(X_ori, y_ori)):

        # a. Init loss, optimizer and params
        loss_fn =  get_loss(args.loss, int(y), 0, args.is_softmax, args.softmax_dim)
        optim = get_optimizer(args.optimizer, model, loss_fn, device)
        params = get_optimization_params(optim, X, args)

        # b. DO run
        start_time = time.time()
        with torch.no_grad():
            new_x, losses, out_list = optim.run(**params)
            end_time = time.time() - start_time

            loss_list.append(losses[-1])
            time_t.append(end_time)

            new_out = model(new_x.view(1, *list(new_x.shape)))
            if not args.is_softmax:
                softmax = nn.Softmax(dim=args.softmax_dim)
                new_out = softmax(new_out)
            new_y = torch.argmax(new_out)

            if int(new_y) != int(y):
                success.append(1)
            else:
                success.append(0)

            if args.L_type == 2:
                l2_dist = torch.norm(X-new_x.cpu())
                if float(l2_dist) > 0:
                    epsilon.append(l2_dist)
            elif args.L_type == -1:
                linf_dist = torch.max(torch.abs(X-new_x.cpu()))
                epsilon.append(linf_dist)

    return success, loss_list, epsilon, time_t


def target_run(args, model, X_ori, y_ori, device):
    success = []
    loss_list = []
    epsilon = []
    time_t = []
    for X, y in tqdm(zip(X_ori, y_ori)):
        for i in range(10):
            if y == i:
                continue
            # a. Init loss, optimizer and params
            loss_fn =  get_loss(args.loss, i, 1, args.is_softmax, args.softmax_dim)
            optim = get_optimizer(args.optimizer, model, loss_fn, device)
            params = get_optimization_params(optim, X, args)

            # b. DO run
            start_time = time.time()
            with torch.no_grad():
                new_x, losses, out_list = optim.run(**params)
                end_time = time.time() - start_time

                loss_list.append(losses[-1])
                time_t.append(end_time)

                new_out = model(new_x.view(1, *list(new_x.shape)))
                if not args.is_softmax:
                    softmax = nn.Softmax(dim=args.softmax_dim)
                    new_out = softmax(new_out)
                new_y = torch.argmax(new_out)

                if int(new_y) == i:
                    success.append(1)
                else:
                    success.append(0)

                if args.L_type == 2:
                    l2_dist = torch.norm(X-new_x.cpu())
                    if float(l2_dist) > 0:
                        epsilon.append(l2_dist)
                elif args.L_type == -1:
                    linf_dist = torch.max(torch.abs(X-new_x.cpu()))
                    epsilon.append(linf_dist)

    return success, loss_list, epsilon, time_t


def main_body(args, device):

    # 1. Get dataloader
    print('3. Loading data...')
    root = '{}{}'.format(args.root, args.data)
    dataloader, train, test = load_data(args.data, args.use_inception, root, args.transform, args.data_batch_size, args.num_workers)


    print('4. Loading model...')
    # 2. Loading the network
    if args.model_path != 'none':
        try:

            net = torch.load(args.model_path).to(device)
        except:
            print('Couldnt load the model from path {}. Either wrong path or model class not found in src/models.')
            print('Using default model for given dataset')
            model = get_model(args.data).to(device)
    else:
        model = get_model(args.use_inception, args.data, device).to(device)
    model.eval()

    # 3. Get data
    X_ori, y_ori = get_data(args.n_example, args.data_batch_size, test, device)

    print('5. Starting the evaluation')
    # Perform run
    if not args.maximise:
        success, loss, epsilon, time = untarget_run(args, model, X_ori, y_ori, device)
    else:
        success, loss, epsilon, time = target_run(args, model, X_ori, y_ori, device)

    return success, loss, epsilon, time



if __name__ == '__main__':

    # 0. Set sys and import
    import sys
    import os
    import time
    import argparse
    import warnings
    import json
    from datetime import datetime
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
    import dataset as data
    import zeroOptim
    import ZOOptim
    import FWOptim



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
    parser.add_argument('--data_batch_size',
                        type=int,               default=50,
                        help='Batch size loading the data. Default 50')
    parser.add_argument('--num_workers',
                        type=int,               default=0,
                        help='Number of workers for the dataloader. Default 0')
    parser.add_argument('--n_example',
                        type=int,               default=100,
                        help='Number of image used for the evaluation. Default is 100')
    # 1.c) Model args
    parser.add_argument('--use_inception',
                        action='store_true',    default=False,
                        help='Tell to use the InceptionV3 model')
    parser.add_argument('--model_path',
                        type=str,               default='none',
                        help='Path of the model to attack')
    # 1.d) Loss args
    parser.add_argument('--loss',
                        type=str,               default='Zoo',
                        help='The type of loss. Either "Zoo" or "MSE". Default is "Zoo"')
    parser.add_argument('--maximise',
                        type=int,               default=0,
                        help='Tell to perform an untargeted attack (0) or targeted(1). Default is 0')
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
                        type=tuple,             default=(0, 1),
                        help='Iterable with the minimum and the maximum value of a pixel. Default is (0, 1)')
    parser.add_argument('--L_type',
                        type=int,               default=-1,
                        help='The norm type. -1 equals infinity notm. Default -1')
    parser.add_argument('--epsilon',
                        type=float,             default=0.5,
                        help='The upper bound of L_infinity or L2 norm')
    parser.add_argument('--verbose',
                        type=int,               default=0,
                        help='The level of verbose. 0 no verbose, 1 partial, 2 total. Default is 0')
    parser.add_argument('--epochs',
                        type=int,               default=100,
                        help='The maximum number of steps')
    parser.add_argument('--max_time',
                        type=float,             default=1000,
                        help='The maximum number of seconds allowed to perform the attack')
    parser.add_argument('--tqdm_disabled',
                        type=int,               default=1,
                        help='If 1 the tqdm bar during the optimization procedure will not be displayed. Default is 1')
    # 1.d) General arguments in Balasubramanian Algortihms
    parser.add_argument('--v',
                        type=float,             default=0.001,
                        help='The gaussian smoothing parameters. Usually the lesser the more precise is the gradient. Default is 0.001')
    parser.add_argument('--n_gradient',
                        type=int,               default=1000,
                        help='The number of gaussian vector generated  for computing the pseudo gradient. Warning: first cause of OutOfMemory. Default 1000')
    parser.add_argument('--batch_size',
                        type=int,               default=-1,
                        help='Batch size during gradient estimation. Default is -1 (= args.n_gradient)')
    # 1.e) Classic ZSCG args
    parser.add_argument('--alpha',
                        type=float,             default=0.2,
                        help='The momentum/lr of Classic ZSCG. Must be in (0, 1). Suggested is 1/sqrt(EPOCH). Default is 0.2. Only for Classic ZSCG')
    # 1.f) Inexact ZSGG args
    parser.add_argument('--gamma',
                        type=float,             default=2,
                        help='Gradient update rule inside ICG. Suggested is 2L. Only for Inexact ZSCG')
    parser.add_argument('--mu',
                        type=float,             default=1/400,
                        help='The first stopping criterion inside ICG. Only for Inexact ZSCG')
    parser.add_argument('--max_t',
                        type=int,               default=50,
                        help='The maximum number of steps inside ICG. Default is 50. Only for Inexact ZSCG')
    # 1.g) Zero SGD args
    parser.add_argument('--lr',
                        type=float,             default=0.1,
                        help='The learning rate for the Zero SGD. Default is 0.01. Only for Zero SGD')
    # 1.h) ZOOptim args
    parser.add_argument('--beta_1',
                        type=float, default=0.9,
                        help='ADAM parameter')
    parser.add_argument('--beta_2',
                        type=float, default=0.999,
                        help='ADAM parameter')
    parser.add_argument('--solver',
                        type=str, default='adam',
                        help='Either ADAM or Newton')
    parser.add_argument('--c',
                        type=float, default=1,
                        help='Hinge-loss wight')
    parser.add_argument('--stop_criterion',
                        action='store_true', default=True,
                        help='Stop if the loss does not decrease for 20 epochs')
    # 1.i) FrankWolfe parameters
    parser.add_argument('--FW_delta',
                        type=float, default=0.001,
                        help='Gaussian smoothing in Frank-Wolfe')
    parser.add_argument('--FW_beta',
                        type=float, default=0.8,
                        help='Momentum at every step')
    parser.add_argument('--FW_gamma',
                        type=float, default=0.2,
                        help='learning_rate')
    # 1.j) Logs Parameters
    parser.add_argument('--logs_path',
                        type=str,             default='../logs',
                        help='The path where to save the run logs')

    args = parser.parse_args()
    print('1. Arguments are:\n\n{}\n'.format(args))

    # 2. Set device and sees
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('2. Device is: {}'.format(device))

    success, loss, epsilon, time_taken = main_body(args, device)
    print('6. Results:')
    print('\tSuccess rate:    {:.2f}%'.format(100*float(np.mean(success))))
    print('\tAverage time:    {:.4f}'.format(float(np.mean(time_taken))))
    print('\tAverage epsilon: {:.2f}'.format(float(np.mean(epsilon))))

    # 10. Save logs
    logs = {'Dataset': args.data,
            'Inception': args.use_inception,
            'Optimizer': args.optimizer,
            'Loss': args.loss,
            'L norm': args.L_type,
            'Target attack': args.maximise,
            'Results': {'Success': float(np.mean(success)),
                        'Mean time': float(np.mean(time_taken)),
                        'Mean epsilon': float(np.mean(epsilon))},
            'All_config_args': vars(args)}

    now = datetime.now()
    log_time = now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
    if args.optimizer == 'zoo':
        LOG_PATH = '{}/logs_{}_{}_{}_{}_{}'.format(args.logs_path, args.optimizer, args.data, args.c, args.L_type, log_time)
    else:
        LOG_PATH = '{}/logs_{}_{}_{}_{}_{}'.format(args.logs_path, args.optimizer, args.data, args.epsilon, args.L_type, log_time)
    print('7. Saving log file and figure at {}.json'.format(LOG_PATH))
    with open('{}.json'.format(LOG_PATH), 'w') as file:
        json.dump(logs, file)
