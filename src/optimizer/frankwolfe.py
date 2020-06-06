"""
Implements Frank Wolfe zeroth order optimization for either white
and black box attacks
"""


# Dependencies
from .optimizer import Optimizer
import numpy as np
import torch
import tqdm


class FrankWolfeBlackBox(Optimizer):

    # Override run function
    def run(
        self, x, m_weight=0.3, step_size=1e-4, l_bound=1e-3, num_epochs=10,
        grad_est='euclidean', grad_est_step=1e-3, grad_est_niter=1e-3,
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
        ret_out (boolean):      Defines wether to compute and return list of losses
                                and outputs at the end of every epoch;
        verbose (float):        Wether to print output or not;
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
        elif grad_est == 'gaussian': # Case estimation function is gaussian
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
        # Loop through each epoch
        for e in tqdm(range(num_epochs), disable=(not verbose)):
            # Compute step
            x_prev, m_prev = self.step(
                x_prev, x_ori, m_prev,
                m_weight=m_weight,
                grad_fn=grad_fn,
                verbose=verbose
            )
            # Case output must be returned: compute it along with loss
            if ret_out or verbose:
                # Compute out
                out = self.model(x_prev)
                # Compute loss
                loss = self.loss(out)
                # Eventually store computed output and loss
                if ret_out:
                    out_list.append(out.squeeze().tolist())
                    loss_list.append(loss.squeeze().item())
                # Eventually print current output and current loss
                if verbose:
                    print('Epoch nr %d' % e+1)
                    print('Current output: %s' % out.squeeze().tolist())
                    print('Current loss: %.03f' % loss.squeeze().item())
                    print()
        # Return new x, loss list, out list
        if ret_out:
            return x_prev, loss, out
        # Return only new x
        return x_prev

    # Override step function
    def step(
        self, x, x_ori, m, grad_fn, m_weight=.3, step_size=1e-4, l_bound=1e-3,
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
        print(grad_fn)  # DEBUG
        print(q_curr)  # DEBUG
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
        u /= torch.norm(u, axis=1).view(-1, 1)
        # Loop through each step
        for i in range(num_iter):
            # Reshape current u
            u_curr = u[i].view(q.shape)
            # Compute output
            out = self.model(torch.cat([
                # 1st item is leftmost term
                x + step_size * u_curr,  # u[i].shape = (1, 1, 28, 28)
                # 2nd item is rightmost term
                x - step_size * u_curr
            # Conactenate over batch size
            ], axis=0))
            # Compute loss
            loss = self.loss(out)
            # Compute update
            q = q + (d / (2 * step_size * num_iter)) * (loss[0] - loss[1]) * u_curr
        # Return estimated gradient
        return q

    # Gaussian estimation of the gradient
    def grad_est_gaussian(self, x, step_size=1e-3, num_iter=10):
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
        # Loop through each step
        for i in range(num_iter):
            # Reshape current u
            u_curr = u[i].view(q.shape)
            # Compute output
            out = self.model(torch.cat([
                # 1st item is leftmost term
                x + step_size * u[i],
                # 2nd item is rightmost term
                x - step_size * u[i]
            ], axis=0))
            # Compute loss
            loss = self.loss(out)
            # Compute update
            q = q + (1 / (2 * step_size * num_iter)) * (loss[0] - loss[1]) * u_curr
        # Return estimated gradient
        return q

    # Compute Linear Minimization Oracle
    def LMO(self, x_ori, m_curr, epsilon=1e-3):
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
        return x_ori - epsilon * torch.sign(m_curr)
