import torch
import numpy as np
from tqdm import tqdm


class ZOOptim(object):

    def __init__(self, model, loss, device='cuda'):
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
    def run(self, x, c, learning_rate=1e-2, n_gradient=128,
            h=1e-4, beta_1=0.9, beta_2=0.999, solver="adam", verbose=False,
            max_steps=10000, stop_criterion=True, epsilon=1e-8,
            tqdm_disable=False, additional_out=False):
        """
        Args:
            Name                    Type                Description
            x:                      (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
            c:                      (float)             Loss weight
            learning_rate:          (float)             Learning rate
            n_gradient:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Gradient estimation accuracy O(h^2)
            beta_1:                 (float)             ADAM hyper-parameter
            beta_2:                 (float)             ADAM hyper-parameter
            solver:                 (str)               Either "ADAM" or "Newton"
            epsilon:                (float)             Avoid dividing by 0
            max_steps:              (int)               The maximum number of steps
            stop_criterion:         (boolean)           If true stop when the loss is 0
            verbose:                (int)               Display information. Default is 0
            tqdm_disable:           (bool)              Disable the tqdm bar. Default is False
            additional_out:         (bool)              Return also all the x. Default is False
        return:
            x:                      (torch.tensor)      Final image
            losses:                 (list)              Loss at each step
            outs:                   (list)              Adversarial example at each step
            l2_dists:               (list)              L2-dist at each step
            losses_st:              (list)              Second loss term at each step
        """

        # Verify Solver is implemented
        if solver.lower() not in ["adam", "newton"]:
            raise NotImplementedError("Unknown solver, use 'adam' or 'newton'")

        # 1. Initialize paramters
        total_dim = int(np.prod(x.shape))
        x_dim = x.shape
        # Scale and Reshape x to be column vector
        x = x.reshape(-1, 1)
        # Store original image
        x_0 = x.clone().detach()
        # Init list of losses (made up of two terms) and outputs for results
        losses, losses_st, l2_dists, outs = [], [], [], []
        # Best Values
        best_l2 = np.inf
        best_image = torch.zeros_like(x)
        best_loss = np.inf
        # Adversarial attack found flag
        found = False

        # Set ADAM parameters
        if solver == "adam":
            self.M = torch.zeros(x.shape).to(self.device)
            self.v = torch.zeros(x.shape).to(self.device)
            self.T = torch.zeros(x.shape).to(self.device)

        # 2. Main Iteration
        for iteration in tqdm(range(max_steps), disable=tqdm_disable):

            # 2.1 Call the step
            x = self.step(x, x_0, c, learning_rate, n_gradient, h, beta_1, beta_2,
                          solver, epsilon, x_dim, total_dim, verbose)

            # 2.2 Compute new loss and store current info
            out = self.model(x.view(1, x_dim[0], x_dim[1], x_dim[2]))
            curr_loss_st = self.loss(out)
            curr_l2 = torch.norm(x - x_0).detach()
            curr_loss = curr_l2 + c * curr_loss_st

            # 2.3 Flag when a valid adversarial example is found
            if curr_loss_st == 0:
                found = True

            # 2.4 Keep best values
            if found:
                # First valid example
                if not losses_st or losses_st[-1] != 0:
                    print("First valid image found at iteration {} with l2-distance = {}".format(iteration, curr_l2))
                    best_l2 = curr_l2
                    best_image = x
                    best_loss = curr_loss
                # New best example
                if curr_l2 < best_l2 and curr_loss_st == 0:
                    best_l2 = curr_l2
                    best_image = x
                    best_loss = curr_loss
                # Worst example
                else:
                    curr_l2 = best_l2
                    curr_loss = best_loss
                    x = best_image
                    curr_loss_st = torch.zeros(1)

            # 2.5 Save results
            outs.append(float(out.detach().cpu()[0, self.loss.neuron].item()))
            losses_st.append(float(curr_loss_st.detach().cpu().item()))
            l2_dists.append(curr_l2)
            losses.append(curr_loss)

            # 2.6 Display  info
            if verbose:
                print('-----------STEP {}-----------'.format(iteration))
                print('Iteration: {}'.format(iteration))
                print('Shape of x: {}'.format(x.shape))
                print('Second term loss:    {}'.format(curr_loss_st.cpu().item()))
                print('L2 distance: {}'.format(curr_l2))
                print("Adversarial Loss:    {}".format(curr_loss))

            # 2.7 Evaluate stop criterion
            if stop_criterion:
              if len(losses) > 20:
                if curr_loss_st == 0 and l2_dists[-1] == l2_dists[-20]:
                    break

        # 3. Unsuccessful attack
        if losses_st[-1] > 0:
            print("Unsuccessful attack")

        # Return
        best_image = best_image.reshape(x_dim[0], x_dim[1], x_dim[2])
        if additional_out:
            return best_image, losses, l2_dists, losses_st, outs
        return best_image, losses, outs


    """
    Do an optimization step
    """
    def step(self, x, x_0, c, learning_rate, n_gradient, h, beta_1, beta_2,
             solver, epsilon, x_dim, total_dim, verbose):
        """
        Args:
            Name                    Type                Description
            x:                      (torch.tensor)      Linearized current adversarial image
            x_0:                    (torch.tensor)      Linearized original image
            c:                      (float)             Loss weight
            learning_rate:          (float)             Learning rate
            n_gradient:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Gradient estimation accuracy O(h^2)
            beta_1:                 (float)             ADAM hyper-parameter
            beta_2:                 (float)             ADAM hyper-parameter
            solver:                 (str)               Either "ADAM" or "Newton"
            epsilon:                (float)             Avoid dividing by 0
            x_dim:                  (torch.size)        Size of the original image
            total_dim:              (int)               Total number of pixels
            verbose:                (int)               Display information. Default is 0
        return:
            x:                      (torch.tensor)      Adversarial example
        """

        # 1. Randomly pick batch_size coordinates
        if n_gradient > total_dim:
            raise ValueError("Batch size must be lower than the total dimension")
        indices = np.random.choice(total_dim, n_gradient, replace=False)
        e_matrix = torch.zeros(n_gradient, total_dim).to(self.device)
        for n, i in enumerate(indices):
            e_matrix[n, i] = 1

        # 2. expand x and x_0 to have n_gradient rows
        x_expanded = x.view(-1).expand(n_gradient, total_dim).to(self.device)
        x_0_expanded = x_0.view(-1).expand(n_gradient, total_dim).to(self.device)

        # 3. Optimizers
        if solver == "adam":
            # 3.1 Gradient approximation
            g_hat = self.compute_gradient(x_0_expanded, x_expanded, e_matrix, c, n_gradient, h, solver, x_dim, verbose).view(-1, 1)

            # 3.2 ADAM Update
            self.T[indices] = self.T[indices] + 1
            self.M[indices] = beta_1 * self.M[indices] + (1 - beta_1) * g_hat
            self.v[indices] = beta_2 * self.v[indices] + (1 - beta_2) * g_hat ** 2
            M_hat = torch.zeros_like(self.M).to(self.device)
            v_hat = torch.zeros_like(self.v).to(self.device)
            M_hat[indices] = self.M[indices] / (1 - beta_1 ** self.T[indices])
            v_hat[indices] = self.v[indices] / (1 - beta_2 ** self.T[indices])
            delta = -learning_rate * (M_hat / (torch.sqrt(v_hat) + epsilon))
            x = x + delta.view(-1, 1)

            # 3.3 Call verbose
            if verbose > 1:
                print('-------------ADAM------------')
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('Chosen indices are: {}\n'.format(indices))
                print('T = {}'.format(self.T))
                print('M = {}'.format(self.M))
                print('v = {}'.format(self.v))
                print('delta = {}'.format(delta))

            # 3.4 Return
            return self.project_boundaries(x).detach()

        elif solver == "newton":
            # 3.1 Gradient and Hessian approximation
            g_hat, h_hat = self.compute_gradient(x_0_expanded, x_expanded, e_matrix, c, n_gradient, h, solver, x_dim, verbose)
            g_hat = g_hat.view(-1, 1)
            h_hat = h_hat.view(-1, 1)

            # 3.2 Update
            delta = torch.zeros(x.shape).to(self.device)
            h_hat[h_hat <= 0] = 1
            delta[indices] = -learning_rate * (g_hat / h_hat)
            x + delta.view(-1, 1)

            # 3.3 Call verbose
            if verbose > 1:
                print('------------NEWTON-----------')
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('Chosen indices are: {}\n'.format(indices))
                print('delta = {}'.format(delta))

            # 3.4 Return
            return self.project_boundaries(x).detach()


    """
    Compute Gradient and Hessian
    """
    def compute_gradient(self, x_0_expanded, x_expanded, e_matrix, c, n_gradient, h, solver, x_dim, verbose):
        """
        Args:
            x_0_expanded:           (torch.tensor)      (n_pixels, n_batches) containing n_batches times the original image
            x_expanded:             (torch.tensor)      (n_pixels, n_batches) containing n_batches times the original image
            e_matrix:               (torch.tensor)      (n_pixels, n_batches)
            c:                      (float)             Loss weight
            n_gradient:             (int)               Coordinates we simultaneously optimize
            h:                      (float)             Gradient estimation accuracy O(h^2)
            solver:                 (str)               Either "ADAM" or "Newton"
            x_dim:                  (torch.size)        Size of the original image
            verbose:                (int)               Display information. Default is 0
        return:
            g_hat:                  (torch.tensor)      Gradient approximation
            h_hat:                  (torch.tensor)      Hessian approximation (if solver=="Newton")
        """
        # 1. Intermediate steps
        input_plus = x_expanded + h * e_matrix
        input_minus = x_expanded - h * e_matrix
        out_plus = self.model(input_plus.view(n_gradient, *list(x_dim)))
        out_minus = self.model(input_minus.view(n_gradient, *list(x_dim)))
        loss2_plus = self.loss(out_plus)
        loss2_minus = self.loss(out_minus)
        loss1_plus = torch.norm(input_plus - x_0_expanded, dim=1)
        loss1_minus = torch.norm(input_minus - x_0_expanded, dim=1)
        first_term = loss1_plus + (c * loss2_plus)
        second_term = loss1_minus + (c * loss2_minus)

        # 2. Compute gradient
        g_hat = (first_term - second_term) / (2 * h)

        # 3. Display info
        if verbose > 0:
            print('-----COMPUTING GRADIENT-----')
            print('g_hat has shape {}'.format(g_hat.shape))
            print('g_hat ='.format(g_hat))

        # 4. Compute hessian
        if solver == "newton":
            loss1_added = torch.norm(x_expanded - x_0_expanded, dim=1)
            loss2_added = self.loss(self.model(x_expanded.view(n_gradient, *list(x_dim))))
            additional_term = loss1_added + (c * loss2_added)
            h_hat = (first_term + second_term - 2 * additional_term) / (h ** 2)

            # 4.1 Display info:
            if verbose > 0:
                print('h_hat has shape {}'.format(h_hat.shape))
                print('h_hat ='.format(h_hat))
            return g_hat.detach(), h_hat.detach()

        return g_hat.detach()


    """
    Check the boundaries of our constraint optimization problem
    """
    def project_boundaries(self, x):
        """
        Args:
            Name              Type                Description
            x:                (torch.tensor)      Linearized current adversarial image
        """

        x[x > 1] = 1
        x[x < 0] = 0
        return x



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
    optim = ZOOptim(model=net, loss=loss)
    x = torch.rand(3, 28, 28)
    optim.run(x, c=0.1, learning_rate=0.1)
