import torch

"""
Classic version of the Zero-order Stochastic Gradient method (ZeroSGD)
"""
class ZeroSGD(object):
    """
    Args:
    Name            Type                Description
    model:          (nn.Module)         The model to use to get the output
    loss:           (nn.Module)         The loss to minimize
    device:
    """
    def __init__(self, model, loss, device=torch.device('cuda')):
        self.device = device
        self.loss = loss
        self.model = model.to(self.device)
        self.model.eval()


    """
    Perform a zero-order optimization
    """
    def run(self, x, v, mk, ak, epsilon, max_steps=100, stop_criterion = 0, max_aux_step = 100, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem
        v:              (float)             The gaussian smoothing
        mK:             (list)              A list of the the number of normal vector to generate at every step
        aK:             (list)              The momentum to use at every step
        epsilon:        (float)             The upper bound of the infinity norm
        max_steps:      (int)               The maximum number of steps
        stop_criterion  (float)             TODO
        max_aux_step    (int)               TODO
        verbose:        (bool)              Display information or not. Default is 0
        """
        self.total_dim = x.shape[0]*x.shape[1]*x.shape[2]
        self.dim  = x.shape
        self.mins = x.clone() - epsilon
        self.max  = x.clone() + epsilon
        self.epsilon = epsilon
        x = x.reshape(1, self.dim[0], self.dim[1], self.dim[2])

        losses, xs, outs = [], [], []
        G, u = [], []
        for ep in range(max_steps):
            x, Gk, uk = self.step(x, v, mk[ep], ak[ep], verbose)
            x = x.reshape(1, self.dim[0], self.dim[1], self.dim[2])
            out  = self.model(x)
            loss = self.loss(out)
            G.append(Gk)
            u.append(uk)
            outs.append(out.cpu()[0, self.loss.neuron].item())
            losses.append(loss.cpu().item())
            xs.append(x.cpu().detach().numpy())
            if verbose:
                print('---------------------------')
                print('Step number: {}'.format(ep))
                print('Shape of x:  {}'.format(x))
                print('New loss:    {}'.format(loss.cpu().item()))
        return xs, losses, [outs, G, u]

    """
    Do an optimization step
    """
    def step(self, x, v, mk, ak, verbose=0):
        # 1 .Create x(k-1) and add a first dimension indicating that we are using an input with batch size equals 1
        x = x.float().to(self.device)
        # 2.Create x(k-1) + v*u(k-1)
        uk     = self.generate_uk(mk)                                                   # Dim (mk, channel*width*height)
        img_u  = uk.reshape(mk, self.dim[0], self.dim[1], self.dim[2]).to(self.device)  # Dim (mk, channel, width, height)
        img_x  = x.expand(mk, self.dim[0], self.dim[1], self.dim[2])                    # Dim (mk, channel, width, height)
        m_x    = (img_x + v*img_u)                                                      # Dim (mk, channel, width, height)

        if verbose > 1:
            print('INPUT')
            print('The Gaussian vector uk has shape:{}'.format(uk.shape))
            print('The input x has shape:\t\t{}'.format(x.shape))
            print('The input x + vu has shape:\t{}'.format(m_x.shape))

        # 3. Get objective functions
        standard_loss = self.compute_loss(x)                                            # Dim (1)
        gaussian_loss = self.compute_loss(m_x)                                          # Dim (mk)
        # 4. Compute gradient approximation
        Gk  = self.compute_Gk(standard_loss, gaussian_loss, v, uk)                       # Dim (channel*width*height)

        if verbose > 1:
            print('OUTPUT')
            print('F(x) has shape:\t\t\t{}'.format(standard_loss.shape))
            print('F(x + vu) has shape:\t\t{}'.format(gaussian_loss.shape))
            # 4. Evaluate approximation of the gradient
            print('GRADIENT APPROXIMATION')
            print('G(u;v,k,x) has shape:\t\t{}'.format(Gk.shape))

        # 4.Find the argmin
        return x.reshape(-1) - ak*Gk/torch.norm(Gk), Gk, uk

    """
    Generate a random normal vector of size mk
    """
    def generate_uk(self, mk):
        return torch.empty(mk, self.total_dim).normal_(mean=0, std=1).to(self.device)


    """
    Compute objective function (loss) given an input for the model
    """
    def compute_loss(self, x):
        out = self.model(x)
        loss = self.loss(out)
        return loss


    """
    Compute the Gv(x(k-1), chi(k-1), u(k)) in order to compute an approximation of the gradient of f(x(k-1), chi(k-1))
    """
    def compute_Gk(self, standard_loss, gaussian_loss, v, uk, verbose=0):
        """
        Args:
        Name            Type                Description
        standard_loss:  (torch.tensor)      The loss given the input
        gaussian_loss:  (torch.tensor)      The loss given the input with a gaussian vector
        v:              (float)             The gaussian smoothing
        uK:             (torch.tensor)      The random Normal vector
        verbose:        (bool)              Display information or not. Default is 0
        """
        # Compute Gv(x(k-1), chi(k-1), u(k))
        fv = ((gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)             # Dim (mk, 1)
        G =  fv * uk                                                                         # Dim (mk, channel*width*height)
        return torch.mean(G, axis=0)                                                         # Dim (channel*width*height)


if __name__ == '__main__':
    from loss import SpecificSoftmaxMSE
    import matplotlib.pyplot as plt
    from torch import nn
    from tqdm import tqdm

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 3, (2, 2), stride=2, padding=1)
            self.linear = nn.Linear(3, 3)


        def forward(self, x):
            x = self.conv(x)
            return self.linear(x.view(x.shape[0], -1))

    epoch = 50
    m = [50]*epoch
    a = [0.9]*epoch
    v = 1


    net = Net()
    loss = SpecificSoftmaxMSE(neuron=2, y_true=0, dim=1)
    optim = ZeroSGD(model=net, loss=loss)

    x = torch.tensor([1])
    xs, loss_curve, r = optim.run(x.view(1, 1, 1), v, m, a, epsilon=0.5,
                                     max_steps=epoch, stop_criterion = 0,
                                     max_aux_step = 100, verbose=0)
    list_x = [i[0, 0, 0, 0] for i in xs]

    min_, max_ = min(list_x), max(list_x)
    losses = []
    for i in tqdm(range(int(min_-1)*10, int(max_+1)*10)):
        x = torch.tensor([i/10]).to(torch.device('cuda'))
        out = net(x.view(1, 1, 1, 1))
        losses.append(loss(out))
    list_x = [i[0, 0, 0, 0] for i in xs]

    plt.plot([i/10 for i in range(int(min_-1)*10, int(max_+1)*10)], losses, label='Loss curve')
    plt.scatter(list_x, loss_curve, label='Parameters')
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Loss')
    plt.title('Loss function')
    plt.grid()
    plt.show()
