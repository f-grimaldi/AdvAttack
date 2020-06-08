import torch
import numpy as np
import math
from tqdm import tqdm


class ZOOptim(object):

    def __init__(self, model, loss, device='cpu'):
        """
        Args:
            Name            Type                Description
            model:          (nn.Module)         The model to use to get the output
            loss:           (nn.Module)         The loss to minimize
            device:         (str)               Used device ("cpu" or "cuda")
        """
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
            max_iterations=10000, stop_criterion=True, epsilon=1e-8,
            tqdm_disable=False, additional_out=False):
        """
        Args:
            Name                    Type                Description
            x:                      (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
            c:                      (float)             Regularization parameter
            learning_rate:          (float)             Learning rate
            batch_size:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Used in gradient approximation (decrease to increase estimation accuracy)
            beta_1:                 (float)             ADAM hyper-parameter
            beta_2:                 (float)             ADAM hyper-parameter
            solver:                 (str)               ADAM or Newton
            epsilon:                (float)             Parameter for update
            max_iterations:         (int)               The maximum number of steps
            stop_criterion          (boolean)           If true stop when the loss is 0
            hierarchical:           (bool)              If True use hierarchical attack
            importance_sampling:    (bool)              If True use importance sampling
            reset_adam_state:       (bool)              If True reset ADAM state after a valid attack is found
            verbose:                (int)               Display information. Default is 0
            tqdm_disable            (bool)              Disable the tqdm bar. Default is False
        return:
            x:                      (torch.tensor)      Final image
            losses:                 (list)
            outs:                   (list)
        """

        # Verify Solver is implemented
        if solver.lower() not in ["adam", "newton"]:
            raise NotImplementedError("Unknown solver, use 'adam' or 'newton'")

        # Dimension
        total_dim = int(np.prod(x.shape))
        x_dim = x.shape
        # Scale and Reshape x to be column vector
        x = (x * 0.8 + 0.1).reshape(-1, 1)
        # Store original image
        x_0 = x.clone().detach()

        # Init list of losses, distances and outputs for results
        losses, l2_dist, outs = [], [], []

        # Set ADAM parameters
        if solver == "adam":
            M = torch.zeros(x.shape).to(self.device)
            v = torch.zeros(x.shape).to(self.device)
            T = torch.zeros(x.shape).to(self.device)

        # Main Iteration
        for iteration in tqdm(range(max_iterations), disable=tqdm_disable):

            # Call the step
            if solver == "adam":
                x, g, M, v, T = self.step(x, M, v, T, batch_size, beta_1, beta_2, h, learning_rate,
                                          epsilon, solver, x_dim, total_dim, verbose)

            elif solver == "newton":
                # todo
                print("Speravi eh?")

            # Compute new loss
            out = self.model(x.view(1, x_dim[0], x_dim[1], x_dim[2]))
            curr_loss = self.loss(out)
            l2 = torch.norm(x - x_0).detach()

            # Save results
            outs.append(float(out.detach().cpu()[0, self.loss.neuron].item()))
            losses.append(float(curr_loss.detach().cpu().item()))
            l2_dist.append(l2)

            # Display current info
            if verbose:
                print('---------------------------')
                print('Iteration: {}'.format(iteration))
                print('Shape of x: {}'.format(x.shape))
                print('New loss:    {}'.format(curr_loss.cpu().item()))
                print('L2 distance: {}'.format(l2))

            # Evaluate stop criterion
            # Flag if we wanted to minimize the output of a neuron and the prediction is now different
            if stop_criterion:
                condition1 = (int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise==0)
                # Flag if we wanted to maximise the output of a neuron and now the neuron has the greatest activation
                condition2 = (int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise==1)
                if condition1 or condition2:
                    break

        # Return
        x = x.reshape(x_dim[0], x_dim[1], x_dim[2])
        if additional_out:
            return x, losses, l2_dist, outs
        return x, losses, outs

    """
    Do an optimization step
    """

    def step(self, x, M, v, T, batch_size, beta_1, beta_2, h, learning_rate,
             epsilon, solver, x_dim, total_dim, verbose):
        """
        Args:
            x:                      (torch.tensor)      The variable of our optimization problem
            M:                      (torch.tensor)      ADAM parameter
            v:                      (torch.tensor)      ADAM parameter
            T:                      (torch.tensor)      N iterations
            batch_size:             (int)               Compute batch_size gradient for each iteration
            beta_1:                 (float)             ADAM hyper-parameter
            beta_2:                 (float)             ADAM hyper-parameter
            h:                      (float)             Used in gradient approximation
            learning_rate:          (float)             Learning rate
            epsilon:                (float)             Parameter for update
            solver:                 (string)            ADAM or Newton
            x_dim:                  (torch.size)        Original image shape
            total_dim:              (int)               Total number of pixels
            verbose:                (int)               Display information
        return:
            x:                      (torch.tensor)      Update Image
            g_hat:                  (torch.tensor)      Gradient estimation
        """

        # 1. Randomly pick batch_size coordinates
        if batch_size > total_dim:
            raise ValueError("Batch size must be lower than the total dimension")
        indices = np.random.choice(total_dim, batch_size, replace=False)
        e_matrix = torch.zeros(batch_size, total_dim).to(self.device)
        for n, i in enumerate(indices):
            e_matrix[n, i] = 1
        x_expanded = x.view(-1).expand(batch_size, total_dim).to(self.device)

        # 2. Call verbose
        if verbose > 1:
            print('INPUT')
            print('The input x has shape:\t\t{}'.format(x.shape))
            print('Chosen indices are: {}\n'.format(indices))

        # 3. Optimizers
        if solver == "adam":
            # Gradient approximation
            g_hat = self.compute_gradient(x_expanded, e_matrix, batch_size, h, solver, x_dim).view(-1, 1)
            # Update
            T[indices] = T[indices] + 1
            M[indices] = beta_1 * M[indices] + (1 - beta_1) * g_hat
            v[indices] = beta_2 * v[indices] + (1 - beta_2) * g_hat ** 2
            M_hat = torch.zeros_like(M).to(self.device)
            v_hat = torch.zeros_like(v).to(self.device)
            M_hat[indices] = M[indices] / (1 - beta_1 ** T[indices])
            v_hat[indices] = v[indices] / (1 - beta_2 ** T[indices])
            delta = -learning_rate * (M_hat / (torch.sqrt(v_hat) + epsilon))
            # Remove constraints
            x = torch.tan(x * math.pi - (math.pi / 2))
            x = x + delta.view(-1, 1)
            x = (torch.atan(x) + (math.pi / 2)) / math.pi

        elif solver == "newton":
            # Gradient and Hessian approximation
            g_hat, h_hat = self.compute_gradient(x_expanded, e_matrix, batch_size, h, solver, x_dim)
            # TODO

        # 4. Call verbose
        if verbose > 1:
            print('OUTPUT')
            print('g_hat has shape {}'.format(g_hat.shape))
            print('g_hat ='.format(g_hat))

        # 5. Return
        return x.detach(), g_hat.detach(), M, v, T

    """
    Compute Gradient and Hessian
    """

    def compute_gradient(self, x_expanded, e_matrix, batch_size, h, solver, x_dim):
        """
        Args:
            x_expanded:             (torch.tensor)      (n_pixels, n_batches) containing n_batches times the original image
            e_matrix:               (torch.tensor)      (n_pixels, n_batches)
            batch_size:             (int)               Compute batch_size gradient for each iteration
            h:                      (float)             Used in gradient approximation
            solver:                 (string)            ADAM or Newton
            x_dim:                  (torch.size)        Original image shape
        return:
            g_hat:                  (torch.tensor)      Gradient approximation
            h_hat:                  (torch.tensor)      Hessian approximation (if solver=="Newton")
        """
        # Intermediate steps
        first_input = x_expanded + h * e_matrix
        second_input = x_expanded - h * e_matrix
        first_out = self.model(first_input.view(batch_size, *list(x_dim)))
        second_out = self.model(second_input.view(batch_size, *list(x_dim)))
        first_term = self.loss(first_out)
        second_term = self.loss(second_out)

        # Compute gradient
        g_hat = (first_term - second_term) / 2 * h

        # Compute hessian
        if solver == "newton":
            h_hat = (first_term + second_term - 2 * self.loss(x_expanded)) / h ** 2
            return g_hat.detach(), h_hat.detach()
        return g_hat.detach()




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
