import torch
import numpy as np
import math


class ZOOptimizer(object):
    """
    Args:
    Name            Type                Description
    model:          (nn.Module)         The model to use to get the output
    loss:           (nn.Module)         The loss to minimize
    device:
    """
    def __init__(self, model, loss, device='cpu'):
        self.device = torch.device(device)
        self.loss = loss
        self.model = model.to(self.device)
        self.model.eval()


    """
    Perform a zero-order optimization
    """
    def run(self, x, c, learning_rate=0.001, batch_size=128,
            h=0.0001, beta_1=0.9, beta_2=0.999, solver="adam", hierarchical=False,
            importance_sampling=False, reset_adam_state=False, verbose=False,
            max_iterations=10000, stop_criterion=1e-10, epsilon=1e-8,
            tqdm_disable=False):
        """
        Args:
        Name                    Type                Description
        x:                      (torch.tensor)      The variable of our optimization problem- Should be a 3D tensor (img)
        c:                      (float)             Regularization parameter
        learning_rate:          (float)             Learning rate
        maximize:               (bool)              True if the attack is targeted, False otherwise
        batch_size:             (int)               Coordinates we simultaneously optimize
        h:                      (float)             Used in gradient approximation (decrease to increase estimation accuracy)
        beta_1:                 (float)             ADAM hyper-parameter
        beta_2:                 (float)             ADAM hyper-parameter
        solver:                 (string)            ADAM or Newton
        hierarchical:           (bool)              If True use hierarchical attack
        importance_sampling:    (bool)              If True use importance sampling
        reset_adam_state:       (bool)              If True reset ADAM state after a valid attack is found
        epsilon:                (float)             The upper bound of the infinity norm
        max_iterations:         (int)               The maximum number of steps
        stop_criterion          (float)             The minimum loss function
        verbose:                (int)               Display information or not. Default is 0
        tqdm_disable            (bool)              Disable the tqdm bar. Default is False
        """

        # Verify Solver is implemented
        if solver.lower() not in ["adam", "newton"]:
            raise NotImplementedError("Unknown solver, use 'adam' or 'newton'")

        self.c = c
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.solver = solver.lower()
        self.h = h
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.verbose = verbose
        self.epsilon = epsilon
        # Extension
        self.hierarchical = hierarchical
        self.importance_sampling = importance_sampling
        self.reset_adam_state = reset_adam_state

        # Dimension
        self.total_dim = int(np.prod(x.shape))
        self.dim = x.shape
        # Store original image
        x_0 = x.clone()
        # Reshape to column vector
        x = x.reshape(-1, 1)
        # Remove constraints
        x = torch.tan(x*math.pi - (math.pi/2))

        # Init list of losses, distances and outputs for results
        losses, l2_dist, outs = [], [], []

        # Set ADAM parameters
        if self.solver == "adam":
            self.M = torch.zeros(x.shape)
            self.v = torch.zeros(x.shape)
            self.T = torch.zeros(x.shape)

        # Main Iteration
        for iteration in tqdm(range(max_iterations), disable=tqdm_disable):

            # Call the step
            x, g = self.step(x, x_0)

            # Compute new loss
            x = x.reshape(1, self.dim[0], self.dim[1], self.dim[2])
            out = self.model(x)
            loss = self.loss(out)
            l2 = torch.norm(x-x_0)

            # Save results
            outs.append(float(out.detach().cpu()[0, self.loss.neuron].item()))
            losses.append(float(loss.detach().cpu().item()))
            l2_dist.append(l2)

            # Display current info
            if verbose:
                print('---------------------------')
                print('Shape of x: {}'.format(x.shape))
                print('x:  {}'.format(x))
                print('New loss:    {}'.format(loss.cpu().item()))
                print('L2 distance: {}'.format(l2))

            # Evaluate stop criterion

        # Return
        x = x.reshape(self.dim[0], self.dim[1], self.dim[2])
        return x, losses, l2_dist, outs


    """
    Do an optimization step
    """
    def step(self, x, x_0):

        # 1. Randomly pick batch_size coordinates
        if self.batch_size > self.total_dim:
            raise ValueError("Batch size must be lower than the total dimension")
        indices = np.random.choice(self.total_dim, self.batch_size, replace=False)  # return np.ndarray(n_batches)
        e_matrix = torch.zeros(self.batch_size, self.total_dim, )
        for n, i in enumerate(indices):
            e_matrix[n, i] = 1
        x_expanded = x.view(-1).expand(self.batch_size, self.total_dim)

        # 2. Call verbose
        if self.verbose:
            print('INPUT')
            print('The input x has shape:\t\t{}'.format(x.shape))
            print('Chosen indices are: {}\n'.format(indices))

        # 3. Optimizers
        if self.solver == "adam":
            # Gradient approximation
            g_hat = self.compute_gradient(x_expanded, e_matrix)
            # Update
            self.T[indices] = self.T[indices] + 1
            self.M[indices] = self.beta_1 * self.M[indices] + (1 - self.beta_1) * g_hat
            self.v[indices] = self.beta_2 * self.v[indices] + (1 - self.beta_2) * g_hat ^ 2
            M_hat = torch.zeros_like(self.M)
            v_hat = torch.zeros_like(self.v)
            M_hat[indices] = self.M[indices] / (1 - self.beta_1 ^ self.T[indices])
            v_hat[indices] = self.v[indices] / (1 - self.beta_2 ^ self.T[indices])
            delta = -self.learning_rate * (M_hat / (torch.sqrt(v_hat) + self.epsilon))

            x[indices] = x[indices] + delta

        elif self.solver == "newton":
            # Gradient and Hessian approximation
            g_hat, h_hat = self.compute_gradient(x_expanded, e_matrix)
            # TODO

        # 4. Call verbose
        if self.verbose:
            print('OUTPUT')
            print('g_hat has shape {}'.format(g_hat.shape()))
            print('g_hat ='.format(g_hat))

        # 5. Return
        return x.detach(), g_hat.detach()


    """
    Compute Gradient and Hessian
    """
    def compute_gradient(self, x_expanded, e_matrix):
        # Intermediate steps
        """
        Args:
            x_expanded: matrix (n_pixels, n_batches) containing n_batches times the original image
            e_matrix: matrix (n_pixels, n_batches)
        """
        first_input = x_expanded + self.h*e_matrix
        second_input = x_expanded - self.h * e_matrix
        first_input_scaled = (torch.atan(first_input)+(math.pi/2))/math.pi
        second_input_scaled = (torch.atan(second_input) + (math.pi / 2)) / math.pi
        first_out = self.model(first_input_scaled.view(self.batch_size, *list(self.dim)))
        second_out = self.model(second_input_scaled.view(self.batch_size, *list(self.dim)))
        first_term = self.loss(first_out)
        second_term = self.loss(second_out)

        # Compute gradient
        g_hat = (first_term + second_term)/2*self.h

        # Compute hessian
        if self.solver == "newton":
            h_hat = (first_term + second_term - 2*self.loss(x_expanded))/self.h**2
            return g_hat, h_hat
        return g_hat


if __name__ == '__main__':
    from loss import ZooLoss
    from torch import nn
    from tqdm import tqdm

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, (2, 2), stride=1, padding=1)
            self.linear = nn.Linear(3*29*29, 10)


        def forward(self, x):
            x = nn.ReLU()(self.conv(x))
            return nn.Sigmoid()(self.linear(x.view(x.shape[0], -1)))

    epoch = 50
    m = [50]*epoch
    a = [0.9]*epoch
    v = 1


    net = Net()
    loss = ZooLoss(0, 0)
    optim = ZOOptimizer(model=net, loss=loss)
    x = torch.rand(3, 28, 28)
    optim.run(x, c=0.1, learning_rate=0.1)
