from tqdm import tqdm
import numpy as np
import torch

class FWOptim():

    # Constructor
    def __init__(self, model, loss, device=torch.device('cpu')):
        # Save inner attributes
        self.device = device
        self.loss = loss
        # Set model in evaulation mode (no gradient)
        # and move it to chosen device
        self.model = model.eval().to(self.device)

    def run(self, x, b, delta, beta, gamma, epsilon,
            L_type='inf', batch_size = -1,
            C = (0, 1), max_steps = 100, verbose=0,
            additional_out = False, tqdm_disable=False):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        b:              (int)               Number of normal vector to generate at every step
        delta:          (float)             The gaussian smoothing
        beta:           (float)             Mmomentum  every step
        gamma:          (float)             Learning rate
        epsilon:        (float)             The upper bound of the infinity norm or l2 norm
        L_type:         (int)               Either -1 for L_infinity or Lx for Lx. Default is -1
        batch_size:     (int)               The maximum parallelization duting the gradient estimation. Default is -1 (=mk)
        C:              (tuple)             The boundaires of the pixel. Default is (0, 1)
        max_steps:      (int)               The maximum number of steps. Default is 100
        verbose:        (int)               Display information or not. Default is 0
        additional_out  (bool)              Return also all the x. Default is False
        tqdm_disable    (bool)              Disable the tqdm bar. Default is False
        """
        # 1. Init params
        self.dim = x.shape
        self.total_dim = int(np.prod(self.dim))
        self.loss_type = loss_type
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.C = C
        x = x.to(self.device)
        self.x_ori = x.clone()

        # 2. Init results lists
        xs, losses, outs = [], [], []

        # 3. Compute m0
        m = self.gradient(x, b, delta, verbose)

        # 4. Main cycle
        for ep in tqdm(range(max_steps), disable=tqdm_disable):

            # 4.1 Compute new values
            x, m = self.step(x.view(*list(self.dim)), m, b, delta, beta, gamma, verbose)

            # 4.2 Compute new out and loss
            out = self.model(x.view(1, *list(self.dim))).detach()
            loss = self.loss(out).detach()

            # 4.3 Save current results
            if additional_out:
                xs.append(x.cpu().numpy())
            losses.append(loss.cpu().item())
            outs.append(out.cpu())

            # 4.4 Display info
            if verbose > 0:
                print('X with shape {} is {}'.format(x.shape, x))
                print('out with shape {} is {}'.format(out.shape, out))
                print('loss with shape {} is {}'.format(loss.shape, loss))

            # 4.5 Check sop condition
            # Flag if we wanted to minimize the output of a neuron and the prediction is now different
            condition1 = (int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise == 0)
            # Flag if we wanted to maximise the output of a neuron and now the neuron has the greatest activation
            condition2 = (int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise == 1)
            if condition1 or condition2:
                break

        # 5. Return
        if additional_out:
            return x, losses, outs, xs

        return x, losses, outs


    def step(self, x, m, b, delta, beta, gamma, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        m:              (torch.tensor)      The last momentum.
        b:              (int)               Number of normal vector to generate at every step
        delta:          (float)             The gaussian smoothing
        beta:           (float)             Mmomentum  every step
        gamma:          (float)             Learning rate
        verbose:        (int)               Different level of verbose
        """
        # 1. Compute current gradient
        q = self.gradient(x, b, delta, verbose)

        # 2. Update momentum
        m = beta*m + (1-beta)*q

        # 3. Perform LMO
        if self.L_type == -1:
            v = self.x_ori.view(-1) - self.epsilon*torch.sign(m)
        elif self.L_type == 2:
            v = self.x_ori.view(-1) - (self.epsilon*m)/torch.norm(m)
        else:
            if self.L_type == 1:
                raise NotImplementedError
            p = self.L_type
            v = self.x_ori.view(-1) - (self.epsilon*torch.abs(m)**(1/(p-1))/torch.sum(torch.abs(m)**(p/p-1))**(1/p)

        # 4. Update x
        x = x.view(-1) + gamma*(v-x.view(-1))
        x[x < self.C[0]] = self.C[0]
        x[x > self.C[1]] = self.C[1]

        # 5. Display info
        if verbose > 0:
            print('INSIDE STEP')
            print('Q is {}'.format(q))
            print('M is {}'.format(m))
            print('V is {}'.format(v))
            print('D is {}'.format(v-x.view(-1)))

        return x.detach(), m.detach()




    def gradient(self, x, b, delta, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        b:              (int)               Number of normal vector to generate at every step
        delta:          (float)             The gaussian smoothing
        verbose:        (int)               Different level of verbose
        """

        # 1. Init Q
        q = torch.zeros_like(x.view(-1)).to(self.device)                                        # Dim C*W*H

        # 2. Prepare input
        uk = torch.empty((b, *list(x.shape))).normal_(mean=0, std=1).to(self.device)            # Dim b, channel, width, height
        x_exapended = x.expand(b, *list(x.shape))                                               # Dim b, channel, width, height
        x_minus = x_exapended - delta*uk
        x_max = x_exapended + delta*uk

        # 3. Get output
        out_max = self.model(x_max)
        out_min = self.model(x_minus)

        # 4. Get loss
        first_term = self.loss(out_max).detach().view(-1, 1).detach()                   # Dim b, 1
        second_term = self.loss(out_min).detach().view(-1, 1).detach()                  # Dim b, 1

        # 5. Display LV 1 Info
        if verbose > 0:
            print('INSIDE GRADIENT ESTIMATION')
            print('Uk has shape: {}'.format(uk.shape))
            print('X min/plus has shape: {}'.format(x_minus.shape))
            print('Model out has shape: {}'.format(out_max.shape))
            print('First term has shape: {}'.format(first_term.shape))

        # 6. Update Q cycle
        for row in range(b):

            # 6.1 Display LV 2 info
            if verbose > 1:
                print('q has shape: {}'.format(q.shape))
                print('coeff is {:.3f}'.format((1/(2*delta*b))))
                print('main term has shape: {}'.format((first_term[row, :] - second_term[row, :]).shape))
                print('uk[row, :] has shape: {}'.format(uk[row, :].view(-1)))

            # 6.2 Update Q
            q = q + (1/(2*delta*b)) * (first_term[row, :] - second_term[row, :]) * uk[row, :].view(-1)

        # 7. Display LV 1 Info
        if verbose > 0:
            print('Gradient with shape {} is: {}'.format(q.shape, q))

        return q.detach()


if __name__ == '__main__':
    from loss import MSELoss
    from torch import nn
    import matplotlib.pyplot as plt

    class Net(nn.Module):

        def __init__(self, A=1, shift=0):
            super().__init__()
            self.A = A
            self.shift = shift

        def forward(self, x):
            return (torch.sin(self.A*x.view(-1) + self.shift)+1)/2

            net = Net(A=4, shift = 0)

    net = Net(A=4)
    optim = FWOptim(model=net, loss=MSELoss(neuron = 0, maximise=0, is_softmax=True))

    numpy_x = np.arange(-2, 2, step=0.02)
    out = net(torch.tensor(numpy_x).view(-1, 1, 1, 1))
    starting_x = torch.tensor([[[0.23]]])


    x, losses, outs, xs = optim.run(starting_x, b=5, delta=0.001, beta=0.8, gamma=0.2, max_steps=10, epsilon = 0.75,
                                    verbose=0, additional_out = True, C = (-1, 1))


    xs.insert(0, starting_x.view(-1).numpy())
    outs.insert(0, net(starting_x).numpy())
    plt.plot(numpy_x, out.numpy(), label = 'Loss curve')
    plt.plot([float(i) for i in xs], outs, color = 'steelblue', label = 'Steps')
    plt.scatter([float(i) for i in xs], outs, color = 'steelblue', label = 'Steps')
    plt.scatter(float(xs[-1]), outs[-1], color='red', label='End point')
    plt.axvline(-1, color='orange', linestyle = '--', label = 'Boundaries')
    plt.axvline(1, color='orange', linestyle = '--')
    plt.xlabel('Input')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
