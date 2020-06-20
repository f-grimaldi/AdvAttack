from tqdm import tqdm
import numpy as np
import torch

class FrankWolfe(object):
    """
    Frank-Wolfe optimizer for black box adversarial attack
    """

    # Constructor
    def __init__(self, model, loss, device=torch.device('cpu')):
        """
        Constructor

        Args
        model (torch.nn.Module) Neural network against which adversarial attack
                                must be carried out
        loss (loss.Loss)        Loss function used to compare current and
                                target output
        device (torch.Device)   Device on which optimizer must be run, where
                                either neural network weights are stored
        """
        # Save inner attributes
        self.device = device
        self.loss = loss
        # Set model in evaulation mode (no gradient)
        # and move it to chosen device
        self.model = model.eval().to(self.device)

    def run(
        self, x, m_weight, step_size, num_epochs, l_bound=0.3, l_type='inf',
        grad_num_iter=100, grad_smooth=0.001, grad_how='gauss', grad_batch_size=100,
        clip=(0, 1), ret_out=False, verbose=False
    ):
        """
        Args
        x (torch.Tensor)        Input tensor (3d image)
        m_weight (float)        Momentum weight, must be in [0-1] interval (beta)
        step_size (float)       Learning rate (gamma)
        num_epochs (int)        The maximum number of steps
        l_bound (float)         Upper bound of l- norm used (epsilon)
        l_type (int/str)        Type of l- norm which must be used
        grad_num_iter (int)     Number of gradient estimation steps (b)
        grad_smooth (float)     Gradient smoothing coefficient (delta)
        grad_how (str)          How to estimate gradient ('gauss' or 'sphere')
        grad_batch_size (int)   Batch size of gradient computations
        clip (tuple)            Clip pixel values in this interval
        ret_out (bool)          Wether to return all the computed images
        verbose (bool)          Whether to print out verbose log or not
        """
        # # 1. Init params
        dim = x.shape
        # self.loss_type = loss_type
        # self.epsilon = epsilon
        # if batch_size == -1:
        #     self.batch_size = n_gradient
        # else:
        #     self.batch_size = batch_size
        # self.C = C
        #
        # x = x.to(self.device)
        # self.x_ori = x.clone()

        # Initialize input image x and store its original values
        x = x.to(self.device)
        x_ori = x.clone()

        # Initialize results containers
        # xs, losses, outs = [], [], []
        x_list, loss_list, out_list = list(), list(), list()

        # Compute first momentum term (estimated gradient)
        m = self.gradient(
            x,  # Input tensor
            num_iter=grad_num_iter,  # Number of gradient estimation iterations
            batch_size=grad_batch_size,  # Batch size for gradient estimation
            smooth=grad_smooth,  # Smoothing coefficient
            how=grad_how,  # How to compute gradient estimation
            verbose=verbose  # Wether to print verbose output or not
        )

        # Loop through epochs
        for e in tqdm(range(num_epochs), disable=(not verbose)):

            # Compute new x and momentum terms
            x, m = self.step(
                x.view(*list(dim)),  # Input tensor
                x_ori,  # Original input tensor
                m,  # Moemntum tensor
                m_weight=m_weight,
                step_size=step_size,
                l_bound=l_bound,
                l_type=l_type,
                grad_num_iter=grad_num_iter,
                grad_batch_size=grad_batch_size,
                grad_smooth=grad_smooth,
                grad_how=grad_how,
                clip=clip,
                verbose=verbose
            )

            # Compute new out and loss
            out = self.model(x.view(1, *list(dim))).detach()
            loss = self.loss(out).detach()

            # Save current results
            if ret_out:
                x_list.append(x.cpu().numpy())
            loss_list.append(loss.cpu().item())
            out_list.append(out.cpu())

            # Verbose log
            if verbose:
                print('X with shape {} is {}'.format(x.shape, x))
                print('out with shape {} is {}'.format(out.shape, out))
                print('loss with shape {} is {}'.format(loss.shape, loss))

            # Check stopping conditions
            if (((int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise == 0))
                or ((int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise == 1))):
                # Case l-norm is infinity
                if l_type == 'inf':
                    break  # Exit from loop
                # Case l-norm is not infinity
                elif torch.norm(x.view(dim) - x_ori) < l_bound:
                    break  # Exit from loop

        # Return either list of inputs
        if ret_out:
            return x, loss_list, out_list, x_list
        # Return
        return x.view(*list(dim)), loss_list, out_list

    def step(
        self, x, x_ori, m, m_weight, step_size, l_bound, l_type,
        grad_num_iter, grad_batch_size, grad_smooth, grad_how,
        clip=(0, 1), verbose=False
    ):
        """
        Makes a single optimization step

        Args
        x (torch.tensor)        Input variable (3d tensor)
        m (torch.tensor)        Previous momentum term
        m_weight (float)        Momentum weight, must be in [0-1] interval (beta)
        step_size (float)       Learning rate coefficent (gamma)
        l_bound (float)         Upper bound of l- norm used
        l_type (int/str)        Type of l- norm which must be used
        grad_num_iter (int)     Number of gradient estimation steps (b)
        grad_batch_size (int)   Size of the gradient batch
        grad_smooth (float)     Gradient smoothing coefficient (delta)
        grad_how (str)          How to estimate gradient ('gauss' or 'sphere')
        clip (tuple)            Clip pixel values in this interval
        verbose (bool)          Wether to display verbose output or not
        """
        # Compute current gradient
        q = self.gradient(
            x,  # Input vector
            num_iter=grad_num_iter,
            batch_size=grad_batch_size,
            smooth=grad_smooth,
            how=grad_how,
            verbose=verbose
        )
        # Update momentum
        m = m_weight * m + (1-m_weight) * q

        # Compute Linear Minimization Oracle
        v = self.LMO(x_ori, m, l_bound=l_bound, l_type=l_type)

        # Update input tensor x
        x = x.view(-1) + step_size*(v-x.view(-1))
        # Clip values
        x = torch.clamp(x, *clip)

        # Display verbose info
        if verbose:
            print('Optimiziation step')
            print('Q is {}'.format(q))
            print('M is {}'.format(m))
            print('V is {}'.format(v))
            print('D is {}'.format(v-x.view(-1)))

        # Return either new x and new momentum
        return x.detach(), m.detach()

    def gradient(self, x, num_iter, batch_size, smooth, how, verbose=False):
        """
        Compute gradient, by looping b times through estimation iterations

        Args
        x (torch.tensor)    The variable of our optimization problem
        num_iter (int)      Number of iterations for (b)
        batch_size (int)    Size of the gradient batch
        smoothing (float)   The smoothing factor (delta)
        verbose (bool)      Wether to display verbose output or not
        """
        # Initialize q update vector
        q = torch.zeros_like(x.view(-1))  # channels x width x height
        q = q.to(self.device)  # Move to correct device

        # Get dimentions of input tensor
        d = int(np.prod(x.shape))

        # Get number of batches in gradient estimation
        num_batches = num_iter // batch_size
        # # Get batch size
        # bs = self.batch_size

        # Loop throgh every batch
        for i in range(num_batches):

            # Initialize empty sampled vectors
            u_k = torch.empty((batch_size, *list(x.shape)))
            coeff = 1.0  # Initialize coefficient
            # Define sampling method
            if how == 'gauss':  # Gaussian sampling
                u_k = u_k.normal_(mean=0, std=1)
                coeff = (1/(2*smooth*num_iter))
            # Euclidean sphere sampling
            elif how == 'sphere':
                u_k = u_k.normal_(mean=0, std=1)
                u_k = u_k/torch.norm(u_k, dim=0)
                coeff = (d/(2*smooth*num_iter))
            else:  # Not defined
                raise NotImplementedError('Sampling method not defined')
            # Move current batch of sampled vectors to device
            u_k = u_k.to(self.device)
            # uk = torch.empty((bs, *list(x.shape))).normal_(mean=0, std=1).to(self.device)   # Dim bs, channel, width, height

            # Expand input vector along batch size axis
            x_expanded = x.expand(batch_size, *list(x.shape))  # bs x channel x width x height

            # Compute "backward" movement
            x_bw = x_expanded - smooth * u_k
            # Compute "forward" movement
            x_fw = x_expanded + smooth * u_k

            # Compute for both positive and negative terms
            out_bw, out_fw = self.model(x_bw), self.model(x_fw)

            # Compute loss for both positive and negative terms
            loss_bw = self.loss(out_bw).detach().view(-1, 1).detach()  # bs x 1
            loss_fw = self.loss(out_fw).detach().view(-1, 1).detach()  # bs x 1

            # 5. Display LV 1 Info
            if verbose:
                print('Gradeint estimation inner cycle')
                print('Uk has shape: {}'.format(u_k.shape))
                print('X min/plus has shape: {}'.format(x_fw.shape))
                print('Model out has shape: {}'.format(out_fw.shape))
                print('Loss has shape: {}'.format(loss_fw.shape))

            # Update q gradient estimate
            for k in range(batch_size):
                # Show verbose info
                if verbose:
                    print('q_k has shape: {}'.format(q.shape))
                    print('Constant coefficient is {:.3f}'.format(coeff))
                    print('Symmetric difference term shape is: {}'.format((loss_fw[k, :] - loss_bw[k, :]).shape))
                    print('u_k has shape: {}'.format(u_k[k, :].view(-1)))
                # Update q
                q = q + (coeff * (loss_fw[k, :] - loss_bw[k, :])) * u_k[k, :].view(-1)

        # Verbose output
        if verbose:
            print('Gradient with shape {} is: {}'.format(q.shape, q))

        # Return estimated gradient q
        return q.detach()

    def LMO(self, x, m, l_bound, l_type):
        """
        Linear Minimization Oracle

        Args
        x (torch.Tensor)        Original input tensor
        m (torch.Tensor)        Momentum tensor
        loss_bound (float)      Upper bound of l- norm used
        loss_type (int or str)  Type of l- norm which must be used
        """
        if l_type == 'inf':  # l-infinity norm
            return x.view(-1) - l_bound * torch.sign(m)
        elif l_type == 2:  # l-2 norm
            return x.view(-1) - (l_bound*m) / torch.norm(m)
        elif l_type == 1:  # l-1 norm (error)
            raise ValueError('Can not use l-1 norm')
        elif isinstance(l_type, int) and l_type > 0:  # Custom norm
            p = l_type
            return x.view(-1) - torch.sign(m) * (l_bound*torch.abs(m)**(1/(p-1))/torch.sum(torch.abs(m)**(p/p-1))**(1/p))
        # No l-norm defined
        raise NotImplementedError('Can not compute given l- norm')


# # Test
# if __name__ == '__main__':
#
#     # Dependencies
#     from ..loss import MSELoss
#     from torch import nn
#     import matplotlib.pyplot as plt
#
#     # Define new network cls
#     class Net(nn.Module):
#
#         def __init__(self, A=1, shift=0):
#             super().__init__()
#             self.A = A
#             self.shift = shift
#
#         def forward(self, x):
#             return (torch.sin(self.A*x.view(-1) + self.shift)+1)/2
#
#     net = Net(A=4)
#     optim = FWOptim(model=net, loss=MSELoss(neuron = 0, maximise=0, is_softmax=True))
#
#     numpy_x = np.arange(-2, 2, step=0.02)
#     out = net(torch.tensor(numpy_x).view(-1, 1, 1, 1))
#     starting_x = torch.tensor([[[0.23]]])
#
#
#     x, losses, outs, xs = optim.run(starting_x, n_gradient=5, delta=0.001, beta=0.8, gamma=0.2, max_steps=10, epsilon = 0.75,
#                                     verbose=2, additional_out = True, C = (-1, 1))
#
#
#     xs.insert(0, starting_x.view(-1).numpy())
#     outs.insert(0, net(starting_x).numpy())
#     plt.plot(numpy_x, out.numpy(), label = 'Loss curve')
#     plt.plot([float(i) for i in xs], [float(i) for i in outs], color = 'steelblue', label = 'Steps')
#     plt.scatter([float(i) for i in xs], [float(i) for i in outs], color = 'steelblue', label = 'Steps')
#     plt.scatter(float(xs[-1]), outs[-1], color='red', label='End point')
#     plt.axvline(-1, color='orange', linestyle = '--', label = 'Boundaries')
#     plt.axvline(1, color='orange', linestyle = '--')
#     plt.xlabel('Input')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper right')
#     plt.grid()
#     plt.show()
