"""
Implements Frank Wolfe zeroth order optimization for either white
and black box attacks
"""


# Dependencies
from .optimizer import Optimizer
from tqdm import tqdm
import numpy as np
import torch


class FrankWolfe(Optimizer):

    # Override run function
    def run(
        self, x, m_weight=0.3, step_size=1e-4, l_bound=1e-3, num_epochs=10,
        grad_est='gaussian', grad_est_step=1e-3, grad_est_niter=1e-3,
        ret_out=True, verbose=False
    ):
        """
        Computes optimization cycle: performs optimization steps for the chosen
        number of epochs.

        Args
        x (torch.tensor):       Input image of size (height x width x channels);
        m_weight (float):       Weight of the momentum term with respect to the
                                (estimated) gradient. Must be between 0 and 1;
        step_size (float):      Value of the update length performed at every step;
        l_bound (float):        Bound to l-norm;
        num_epochs (int):       Number of loops to execute before returning result;
        grad_est (str):         Name of gradient estimation function
        grad_est_step (float):  Step size to be used in gradient estimation;
        grad_est_niter (float): Number of estimation required in gradient estimation;
        ret_out (bool):      Defines wether to compute and return list of losses
                                and outputs at the end of every epoch;
        verbose (bool):        Wether to print output or not;
        """
        # Initialize losses and output container
        loss_list, out_list = list(), list()
        # Initialize gradient estimation function
        grad_fn = None
        # Choose gradient estimator
        if grad_est == 'euclidean':  # Case estimation function is euclidean
            grad_fn = lambda self, x: self.grad_est_euclidean(
                x=x,
                step_size=grad_est_step,
                num_iter=grad_est_niter
            )
        elif grad_est == 'gaussian':  # Case estimation function is gaussian
            grad_fn = lambda self, x: self.grad_est_gaussian(
                x=x,
                step_size=grad_est_step,
                num_iter=grad_est_niter
            )
        else:  # Case no valid gradient estimator function
            # Raise an error
            raise NotImplementedError('No valid gradient estimator')
        # Initialize original image
        x_ori = x
        # Initialize previous step image
        x_prev = x_ori.clone()
        # Initialize initial momentum
        m_prev = grad_fn(self, x_prev)
        # # DEBUG
        # print('Gradient function is', grad_fn)
        # Loop through each epoch
        for e in tqdm(range(num_epochs), disable=(not verbose)):
            # Compute step
            x_prev, m_prev = self.step(
                x_prev, x_ori, m_prev,
                m_weight=m_weight,
                grad_fn=grad_fn,
                verbose=verbose
            )
            # Print current output
            print('Current x\n', x_prev)
            # Case output must be returned: compute it along with loss
            if ret_out or verbose:
                # Compute out
                out_curr = self.model(x_prev.unsqueeze(0))
                # Compute loss
                loss_curr = self.loss(out_curr)
                # Put everything to CPU
                out_curr, loss_curr = out_curr.cpu(), loss_curr.cpu()
                # Eventually store computed output and loss
                if ret_out:
                    out_list.append(out_curr.squeeze().tolist())
                    loss_list.append(loss_curr.squeeze().item())
                # Eventually print current output and current loss
                if verbose:
                    print('Epoch nr {}'.format(e+1))
                    print('Current output: {}'.format(out_curr.squeeze().tolist()))
                    print('Current loss: {:.03f}'.format(loss_curr.squeeze().item()))
                    print()
        # Return new x, loss list, out list
        if ret_out:
            return x_prev, loss_list, out_list
        # Return only new x
        return x_prev

    # Override step function
    def step(
        self, x, x_ori, m, grad_fn, m_weight=.1, step_size=1e-4, l_bound=0.03,
        verbose=False
    ):
        """
        Computer a single optimization step.

        Args:
        x (torch.tensor):       Input image of size (height x width x channels);
        x_ori (torch.tensor):   Original input image, not modified;
        m (torch.tensor):       Current moment update;
        grad_fn (function):     Function which estimates the gradient;
        m_weight (float):       Weight of the momentum term with respect to the
                                (estimated) gradient. Must be between 0 and 1;
        step_size (float):      Value of the update length performed at every step;
        l_bound (float):        Bound to l-norm;
        verbose (float):        Wether to print output or not;
        """
        # Compute q
        q_curr = grad_fn(self, x)
        # print(grad_fn)  # DEBUG
        # print(q_curr)  # DEBUG
        # Compute new momentum update
        m_curr = m_weight * m + (1-m_weight) * q_curr
        # Computer Linear Minimization Oracle
        v_curr = self.LMO(x_ori, m_curr, l_bound)
        # Compute movement from current x
        d_curr = v_curr - x
        # Make update step (prev x of next step is current x)
        x_curr = x + step_size * d_curr
        # Return either current x and current momentum
        return x_curr, m_curr

    # Euclidean estimation of the gradient
    def grad_est_euclidean(self, x, step_size=1e-3, num_iter=10):
        """
        Samples vectors uniformly from the surface of the Euclidean unit shpere
        Ref: https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere

        Args:
        x (torch.tensor):   Input image (channels x height x width);
        est_size (float):   Step size for gradient estimation;
        num_iter (float):   Number of epochs to estimate gradient;
        """
        # Intiialize estimated gradient with 0 vector (same size as input one)
        q = torch.zeros_like(x)  # x.shape() = (1, 28, 28)
        # Get shape from input vector
        d = np.prod(q.shape)  # Linearized = 1 * 28 * 28
        # Sample <num iter> vectors from euclidean sphere surface
        u = torch.randn(num_iter, d)
        u /= torch.norm(u, dim=1).view(-1, 1)
        # Reshape u such as it has size (num_iter, x.shape)
        u = u.view(num_iter, *list(x.shape))
        # Move u to selected device
        u = u.to(self.device)
        # Expand x along batch size axis
        # We want to have an x item for every u item
        x = x.expand(num_iter, *list(x.shape))
        # Compute network input
        net_in = torch.cat(dim=0, tensors=[
            # 1st half of input batch is for forward terms
            x + step_size * u,
            # 2nd half of input batch is for backward terms
            x - step_size * u
        ])
        # Compute outputs
        net_out = self.model(net_in)
        # Compute losses
        net_loss = self.loss(net_out)
        # # DEBUG
        # print('Input image\n', x[0].view(-1))
        # print('First sampled image is\n', u[0].view(-1))
        # print('First network input image\n', net_in[0].view(-1))
        # print('Check sampled vector norm\n', torch.norm(u[0].view(-1), dim=0))
        # # Check norm of first vector
        # assert torch.norm(u[8].view(-1), dim=0).item() == 1.0, 'Wrong vector normalization'
        # Loop through each step
        for i in range(num_iter):
            # Get forward loss term
            fw = net_loss[i]
            # Get backward loss term
            bw = net_loss[i + num_iter]
            # # Get current u
            # u_curr = u[i].view(-1)
            # Compute update
            q = q + ((d / (2 * step_size * num_iter)) * (fw - bw) * u[i])
        # Return estimated gradient
        return q

    # Gaussian estimation of the gradient
    def grad_est_gaussian(self, x, step_size=1e-3, num_iter=1):
        """
        Samples vectors uniformly from the standard gaussian distribution

        Args:
        x (torch.tensor):   Input image (sizes height x width x channels);
        est_size (float):   Step size for gradient estimation;
        num_iter (float):   Number of epochs to estimate gradient;
        """
        # Intiialize estimated gradient with 0 vector (same size as input one)
        q = torch.zeros_like(x)
        # TODO Get shape from input vector
        d = np.prod(q.shape)
        # Sample <num iter> vectors from euclidean sphere surface
        u = torch.randn(num_iter, d)
        # Reshape u such as it has size (num_iter, x.shape)
        u = u.view(num_iter, *list(x.shape))
        # Move u to selected device
        u = u.to(self.device)
        # Expand x along batch size axis
        # We want to have an x item for every u item
        x = x.expand(num_iter, *list(x.shape))
        # Compute function input terms
        net_in = torch.cat(dim=0, tensors=[
            # 1st half of input batch is for forward terms
            x + step_size * u,
            # 2nd half of input batch is for backward terms
            x - step_size * u
        ])
        # Compute outputs
        net_out = self.model(net_in)
        # Compute losses
        net_loss = self.loss(net_out)
        # Loss output
        for i in range(num_iter):
            # Get forward loss term
            fw = net_loss[i]
            # Get backward loss term
            bw = net_loss[i + num_iter]
            # Compute update
            q = q + (1 / (2 * step_size * num_iter)) * (fw - bw) * u[i]
        # Free memory
        del x, u, net_out, net_loss
        # Return estimated gradient
        return q

    # Compute Linear Minimization Oracle
    def LMO(self, x_ori, m_curr, l_bound=0.05):
        """
        Computes Linear Minimization Oracle, according to some input.
        Here we use L-infinity norm, changin to another norm means changing
        this method either, since solution takes advantage of this specific
        problem settings.

        Args:
        x_ori (torch.Tensor)    Original image (the one from which optimization
                                process started);
        m_curr (torch.Tensor)   Current momentum term;
        epsilon (float)         Bound over L-infinity norm
        """
        return x_ori - l_bound * torch.sign(m_curr)
