import torch
from tqdm import tqdm

"""
Classic version of the Zero-order Stochastic Gradient Descent method (ZeroSGD)
Based on the paper:
    Zeroth-order Nonconvex Stochastic Optimization: Handling Constraints, High-Dimensionality and Saddle-Points∗
by:
    Krishnakumar Balasubramanian†1 and Saeed Ghadimi‡2
ALG.5
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
    def run(self, x, v, mk,
            ak, epsilon,
            batch_size=-1,
            C=(0, 1), verbose=0,
            max_steps=100,
            additional_out=False,
            tqdm_disabled=False):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem- Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        mK:             (list)              A list of the the number of normal vector to generate at every step
        aK:             (list)              The learning rate to use at every step
        epsilon:        (float)             The upper bound of the infinity norm
        batch_size:     (int)               The number of batch size when estimating the gradient. Default -1 (=mk)
        C:              (tuple)             The range of pixel values
        max_steps:      (int)               The maximum number of steps
        verbose:        (int)               Display information or not. Default is 0
        additional_out: (bool)              Return also all the x. Default is False
        tqdm_disabled:  (bool)              Disable the tqdm bar. Default is False
        """

        self.total_dim = x.shape[0]*x.shape[1]*x.shape[2]
        self.dim  = x.shape

        x = x.reshape(1, self.dim[0], self.dim[1], self.dim[2])

        self.mins = (x.clone() - epsilon).view(-1).to(self.device)
        self.max  = (x.clone() + epsilon).view(-1).to(self.device)
        self.epsilon = epsilon
        self.batch = batch_size

        # Init list for results
        losses, outs = [], []  #List of losses and outputs
        xs = []

        # Optimizaion Cycle
        self.x = x
        for ep in tqdm(range(max_steps), disable=tqdm_disabled):

            # Call the step
            x, Gk = self.step(x, v, mk[ep], ak[ep], verbose)

            # Project on boundaries
            x[self.max-x<0] = self.max[self.max-x<0]
            x[x-self.mins<0] = self.mins[x-self.mins<0]
            x[x < C[0]] = C[0]
            x[x > C[1]] = C[1]

            # Compute new loss
            x = x.reshape(1, self.dim[0], self.dim[1], self.dim[2])
            out = self.model(x)
            loss = self.loss(out)

            # Save results
            outs.append(float(out.detach().cpu()[0, self.loss.neuron].item()))
            losses.append(float(loss.detach().cpu().item()))
            if additional_out:
                xs.append(x.detach().cpu())

            # Display current info
            if verbose:
                print('---------------------------')
                print('Step number: {}'.format(ep))
                print('Does X required grad?: {}'.format(x.requires_grad))
                print('Shape of x: {}'.format(x.shape))
                print('Shape of min: {}'.format(self.mins.shape))
                print('Shape of max: {}'.format(self.max.shape))
                print('x:  {}'.format(x))
                print('New loss:    {}'.format(loss.cpu().item()))

            # Evaluate stop criterion
            # Flag if we wanted to minimize the output of a neuron and the prediction is now different
            condition1 = (int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise == 0)
            # Flag if we wanted to maximise the output of a neuron and now the neuron has the greatest activation
            condition2 = (int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise == 1)
            if condition1 or condition2:
                break

        # Return
        x = x.reshape(self.dim[0], self.dim[1], self.dim[2])
        if additional_out:
            return x, losses, outs, xs
        return x, losses, outs

    """
    Do an optimization step
    """
    def step(self, x, v, mk, ak, verbose=0):

        # 1 .Create x(k-1) and add a first dimension indicating that we are using an input with batch size equals 1
        x = x.float().to(self.device)

        # 2.Create x(k-1) + v*u(k-1)
        uk = torch.randn(mk, self.total_dim).to(self.device)                        # Dim (mk, channel*width*height)
        img_u = uk.reshape(mk, self.dim[0], self.dim[1], self.dim[2]).to(self.device)  # Dim (mk, channel, width, height)
        img_x = x.expand(mk, self.dim[0], self.dim[1], self.dim[2])                    # Dim (mk, channel, width, height)
        m_x = (img_x + v*img_u)                                                      # Dim (mk, channel, width, height)

        if verbose > 1:
            print('INPUT')
            print('The Gaussian vector uk has shape:{}'.format(uk.shape))
            print('The input x has shape:\t\t{}'.format(x.shape))
            print('The input x + vu has shape:\t{}'.format(m_x.shape))

        # 3. Get objective functions
        standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))               # Dim (1)
        gaussian_loss = torch.zeros(mk).to(self.device)                               # Dim (mk)
        if self.batch == -1:
            gaussian_loss = self.loss(self.model(m_x))
        else:
            for n in range(mk//self.batch):
                tmp_loss = self.loss(self.model(m_x[n*self.batch:(n+1)*self.batch, :]))
                gaussian_loss[n*self.batch:(n+1)*self.batch] = tmp_loss.detach()        # Dim (mk)

        if verbose > 1:
            print('Standard Loss is: {}'.format(standard_loss))
            print('Gaussian Loss is: {}'.format(gaussian_loss))

        # 4. Compute gradient approximation
        Gk = self.compute_Gk(standard_loss, gaussian_loss, v, uk)            # Dim (channel*width*height)

        # 5. Call verbose
        if verbose > 1:
            print('OUTPUT')
            print('F(x) has shape:\t\t\t{}'.format(standard_loss.shape))
            print('F(x + vu) has shape:\t\t{}'.format(gaussian_loss.shape))
            print('GRADIENT APPROXIMATION')
            print('G(u;v,k,x) has shape:\t\t{}'.format(Gk.shape))
            print('Gradient: {}'.format(ak*Gk/torch.norm(Gk)))

        del uk, standard_loss, gaussian_loss, m_x, img_x, img_u
        # 6. Return
        return (x.reshape(-1) - ak*Gk/torch.norm(Gk)).detach(), Gk.detach()


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
        fv = ((gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)        # Dim (mk, 1)
        G = fv * uk                                                                    # Dim (mk, channel*width*height)
        return torch.mean(G, axis=0)                                                    # Dim (channel*width*height)



"""
Zero-order Stochastic Conditional Gradient (ZSCG)
Based on the paper:
    Zeroth-order Nonconvex Stochastic Optimization: Handling Constraints, High-Dimensionality and Saddle-Points∗
by:
    Krishnakumar Balasubramanian†1 and Saeed Ghadimi‡2
ALG.1
"""
class ClassicZSCG(object):
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
    Perform an attack against the model given an input and the required run params
    """
    def run(self, x, v, n_gradient, ak , epsilon, L_type=-1, batch_size = -1, C = (0, 1),
            max_steps=100, verbose=0, additional_out=False, tqdm_disabled=False):

        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        n_gradient:     (list)              Number of normal vector to generate at every step
        ak:             (list)              Momentum  every step
        epsilon:        (float)             The upper bound of norm
        L_type:         (int)               Either -1 for L_infinity or x for Lx. Default is -1
        batch_size:     (int)               Maximum parallelization during the gradient estimation. Default is -1 (=n_grad)
        C:              (tuple)             The boundaires of the pixel. Default is (0, 1)
        max_steps:      (int)               The maximum number of steps. Default is 100
        verbose:        (int)               Display information or not. Default is 0
        additional_out: (bool)              Return also all the x. Default is False
        tqdm_disabled:  (bool)              Disable the tqdm bar. Default is False
        """

        x = x.to(self.device)
        # 1. Init class attributes
        self.x_original = x.clone()
        self.dim = x.shape
        self.total_dim = torch.prod(torch.tensor(x.shape))
        self.epsilon = epsilon
        self.L_type = L_type
        self.C = C
        self.batch = batch_size


        # 2. Init list of results
        losses, outs = [], []
        x_list = []

        # 3. Main optimization cycle
        for ep in tqdm(range(max_steps), disable=tqdm_disabled):

            if verbose:
                print("---------------")
                print("Step number: {}".format(ep))

            # 3.1 Call the step
            x, gk = self.step(x, v, ak[ep], n_gradient[ep], verbose)
            x = x.reshape(self.dim[0], self.dim[1], self.dim[2]).detach()

            # 3.2 Compute loss
            out = self.model(x.view(1, self.dim[0], self.dim[1], self.dim[2]))
            loss = self.loss(out)

            # 3.3 Save results
            losses.append(loss.detach().cpu().item())
            outs.append(out.detach().cpu()[0, self.loss.neuron].item())
            if additional_out:
                x_list.append(x.cpu())

            # 3.4 Display current info
            if verbose:
                print("Loss:        {}".format(losses[-1]))
                print("Output:      {}".format(outs[-1]))

            # 3.5 Check Stopping criterions
            condition1 = (int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise == 0)
            condition2 = (int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise == 1)
            if condition1 or condition2:
                break

        if additional_out:
            return x, losses, outs, x_list
        return  x, losses, outs


    """
    Do an optimization step
    """
    def step(self, x, v, ak, mk, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        ak:             (float)             The weight avarage in the updating phase. (1 means take only the new x, 0 means no update)
        mk:             (int)               The number of Gaussian Random Vector to generate
        verbose:        (bool)              Display information or not. Default is 0
        """

        # Compute the approximated gradient
        g = self.compute_Gk(x, v, mk, verbose)

        # Call the inexact conditional gradient
        x_g = self.compute_CG(g, verbose).reshape(x.shape[0], x.shape[1], x.shape[2])
        x_new = (1-ak)*x + ak*x_g

        if verbose > 1:
            print("\nINSIDE STEP")
            print("Gradient has shape: {}".format(g.shape))
            print("Gradient is:\n{}".format(g))
            print("x_new has shape: {}".format(x_new.shape))
            print("x_new is:\n{}".format(x_new))

        return x_new.detach(), g.detach()


    """
    Prepare parallelization for the gradient estimation
    """
    def get_parallel(self, x, bs, v):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The current variable
        bs:             (int)               The maximum bacth size
        v:              (int)               The Gaussian smoothing
        """
        uk     = torch.empty(bs, self.total_dim).normal_(mean=0, std=1).to(self.device) # Dim (bs, channel*width*height)
        img_u  = uk.reshape(bs, self.dim[0], self.dim[1], self.dim[2])                  # Dim (bs, channel, width, height)
        img_x  = x.expand(bs, self.dim[0], self.dim[1], self.dim[2])                    # Dim (bs, channel, width, height)
        m_x    = (img_x + v*img_u)

        return m_x, uk

    """
    Compute the Gv(x(k-1), chi(k-1), u(k)) in order to compute an approximation of the gradient of f(x(k-1), chi(k-1))
    """
    def compute_Gk(self, x, v, mk, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        mk:             (int)               The number of Gaussian Random Vector to generate
        verbose:        (bool)              Display information or not. Default is 0
        """

        # 1. Get objective functions
        # CASE BATCH_SIZE == N_GRADIENT
        if self.batch == -1:

            # 1.a Compute standard Loss
            standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))

            # 1.b Compute gaussian loss
            m_x, uk = self.get_parallel(x, mk, v)
            gaussian_loss = self.loss(self.model(m_x))

            if verbose > 1:
                print('\nINSIDE GRADIENT')
                print('The Gaussian vector uk has shape:{}'.format(uk.shape))
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('The input x + vu has shape:\t{}'.format(m_x.shape))

            # 1.c Compute Gv(x(k-1), chi(k-1), u(k))
            fv = ((gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)        # Dim (mk, 1)
            G = fv * uk                                                                     # Dim (mk, channel*width*height)

            return torch.mean(G, axis=0).detach()

        # CASE BATCH_SIZE < N_GRADIENT
        else:

            # 1.a Compute standard loss
            standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))                   # Dim (1)
            G_tot = torch.zeros(mk//self.batch, self.total_dim).to(self.device)                 # Dim (n_batches, hannel*width*height)

            #1.b Compute Gradient
            for n in range(mk//self.batch):
                from_, to_ = n*self.batch, (n+1)*self.batch

                # 1.b Create batch x(k-1) + v*u(k-1)
                m_x, uk = self.get_parallel(x, self.batch, v)

                # 1.c Compute
                tmp_gaussian_loss = self.loss(self.model(m_x)).detach()                                 # Dim(bs)

                if verbose > 1:
                    print('\nINSIDE GRADIENT')
                    print('The Gaussian vector uk has shape:{}'.format(uk.shape))
                    print('The input x has shape:\t\t{}'.format(x.shape))
                    print('The input x + vu has shape:\t{}'.format(m_x.shape))

                # 1.d Compute Gradient
                fv = ((tmp_gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)            # Dim (bs, 1)
                G = fv * uk                                                                             # Dim (bs, channel*width*height)

                if verbose > 1:
                    print('Gaussian cycle loss has shape:\t{}'.format(tmp_gaussian_loss.shape))
                    print('Function approx has shape:\t{}'.format(fv.shape))
                    print('Gradient has shape:\t\t{}'.format(G.shape))

                G_tot[n] = torch.mean(G, axis=0).detach()

        return torch.mean(G_tot, axis=0).detach()


    def compute_CG(self, g, verbose):
        """
        Args:
        Name         Type                Description
        g:           (torch.tensor)      The approximated gradient. Should be a 1D tensor
        """

        # Infinity norm
        if self.L_type == -1:
            x_new = self.x_original.view(-1) - self.epsilon*torch.sign(g)

        # L1 norm
        elif self.L_type == 1:
            raise NotImplementedError

        elif self.L_type == 2:
            x_new = self.x_original.view(-1) - (self.epsilon*g)/torch.norm(g, 2)

        # Generic Lp norm (1 < p < +inf)
        else:
            p = self.L_type
            gp = torch.abs(g)**(1/p-1)
            h = torch.sign(g) * (gp) / torch.norm(gp, p)
            x_new = self.x_original.view(-1) - self.epsilon*h

        # Enforce pixel boundaries
        x_new[x_new < self.C[0]] = self.C[0]
        x_new[x_new > self.C[1]] = self.C[1]

        if verbose > 1:
            print('\nINSIDE CG')
            print('Epsilon * Sign(g) is {}'.format(self.epsilon*torch.sign(g)))
            print('Unchecked new x is: {}'.format(self.x_original.view(-1) - self.epsilon*torch.sign(g)))
            print('The CG gradient is:\n{}'.format(g))
            print('The new x is:\n {}'.format(x_new))

        return x_new.detach()


"""
Zero-order Stochastic Conditional Gradient (ZSCG) with Inexact Conditional Gradient (ICG) update.
Based on the paper:
    Zeroth-order Nonconvex Stochastic Optimization: Handling Constraints, High-Dimensionality and Saddle-Points∗
by:
    Krishnakumar Balasubramanian†1 and Saeed Ghadimi‡2
ALG.4 (modified version for non-convex problem)
"""
class InexactZSCG(object):
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
    Perform an attack against the model given an input and the required run params
    """
    def run(self, x, v, n_gradient, gamma_k, mu_k, epsilon, L_type = -1, batch_size = -1, C = (0, 1),
            max_steps=100, verbose=0, additional_out=False, tqdm_disabled=False, max_t=1000):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        n_gradient:     (list)              Number of normal vector to generate at every step
        gamma_k:        (list)              Momentum at every step inside ICG
        mu_k:           (list)              Stoppinc criterion at every step k inside ICG
        max_t:          (int)               The maximum number of iteration inside of ICG.
        epsilon:        (float)             The upper bound of norm
        L_type:         (int)               Either -1 for L_infinity or x for Lx. Default is -1
        batch_size:     (int)               Maximum parallelization during the gradient estimation. Default is -1 (=n_grad)
        C:              (tuple)             The boundaires of the pixel. Default is (0, 1)
        max_steps:      (int)               The maximum number of steps. Default is 100
        verbose:        (int)               Display information or not. Default is 0
        additional_out: (bool)              Return also all the x. Default is False
        tqdm_disabled:  (bool)              Disable the tqdm bar. Default is False
        """

        x = x.to(self.device)

        # 1. Init class attributes
        self.x_original = x.clone()
        self.dim = x.shape
        self.total_dim = torch.prod(torch.tensor(x.shape))
        self.epsilon = epsilon
        self.L_type = L_type
        self.C = C
        self.batch = batch_size
        self.max_t = max_t

        # 2. Init list of results
        losses, outs = [], [],
        input_list = []

        # 3. Main optimization cycle
        for ep in tqdm(range(max_steps), disable=tqdm_disabled):

            if verbose:
                print("---------------")
                print("Step number: {}".format(ep))

            # 3.1 Call the step
            x, gk = self.step(x, v, gamma_k[ep], mu_k[ep], n_gradient[ep], verbose)
            x = x.reshape(self.dim[0], self.dim[1], self.dim[2]).detach()

            # 3.2 Compute loss
            out = self.model(x.view(1, self.dim[0], self.dim[1], self.dim[2]))
            loss = self.loss(out).view(-1, 1)

            # 3.3 Save results
            losses.append(loss.detach().cpu().item())
            outs.append(out.detach().cpu()[0, self.loss.neuron].item())
            if additional_out:
                input_list.append(x.cpu())

            # 3.4 Display current info
            if verbose:
                print("Loss:        {}".format(losses[-1]))
                print("Output:      {}".format(outs[-1]))

            # 3.5 Check Stopping criterion
            condition1 = (int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise == 0)
            condition2 = (int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise == 1)
            if condition1 or condition2:
                break

        if additional_out:
            return x, losses, outs, input_list
        return x, losses, outs

    """
    Do an optimization step
    """
    def step(self, x, v, gamma, mu, mk, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        gamma:          (float)             The update parameters of g
        mu:             (float)             The stopping criterion
        mk:             (int)               The number of Gaussian Random Vector to generate
        verbose:        (bool)              Display information or not. Default is 0
        """
        # Compute the approximated gradient
        g = self.compute_Gk(x, v, mk, verbose)
        # Call the inexact conditional gradient
        x_new = self.compute_ICG(x, g, gamma, mu, verbose).reshape(x.shape[0], x.shape[1], x.shape[2])

        if verbose > 1:
            print("\nINSIDE STEP")
            print("Gradient has shape: {}".format(g.shape))
            print("Gradient is:\n{}".format(g))
            print("x_new has shape: {}".format(x_new.shape))
            print("x_new is:\n{}".format(x_new))

        return x_new.detach(), g.detach()


    """
    Prepare parallelization for the gradient estimation
    """
    def get_parallel(self, x, bs, v):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The current variable
        bs:             (int)               The maximum bacth size
        v:              (int)               The Gaussian smoothing
        """
        uk     = torch.empty(bs, self.total_dim).normal_(mean=0, std=1).to(self.device) # Dim (bs, channel*width*height)
        img_u  = uk.reshape(bs, self.dim[0], self.dim[1], self.dim[2])                  # Dim (bs, channel, width, height)
        img_x  = x.expand(bs, self.dim[0], self.dim[1], self.dim[2])                    # Dim (bs, channel, width, height)
        m_x    = (img_x + v*img_u)

        return m_x, uk

    """
    Compute the Gv(x(k-1), chi(k-1), u(k)) in order to compute an approximation of the gradient of f(x(k-1), chi(k-1))
    """
    def compute_Gk(self, x, v, mk, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        mk:             (int)               The number of Gaussian Random Vector to generate
        verbose:        (bool)              Display information or not. Default is 0
        """

        # 1. Get objective functions
        # CASE BATCH_SIZE == N_GRADIENT
        if self.batch == -1:

            # 1.a Compute standard Loss
            standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))

            # 1.b Compute gaussian loss
            m_x, uk = self.get_parallel(x, mk, v)
            gaussian_loss = self.loss(self.model(m_x))

            if verbose > 1:
                print('\nINSIDE GRADIENT')
                print('The Gaussian vector uk has shape:{}'.format(uk.shape))
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('The input x + vu has shape:\t{}'.format(m_x.shape))

            # 1.c Compute Gv(x(k-1), chi(k-1), u(k))
            fv = ((gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)        # Dim (mk, 1)
            G = fv * uk                                                                     # Dim (mk, channel*width*height)

            return torch.mean(G, axis=0).detach()

        # CASE BATCH_SIZE < N_GRADIENT
        else:

            # 1.a Compute standard loss
            standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))                   # Dim (1)
            G_tot = torch.zeros(mk//self.batch, self.total_dim).to(self.device)                 # Dim (n_batches, hannel*width*height)

            #1.b Compute Gradient
            for n in range(mk//self.batch):
                from_, to_ = n*self.batch, (n+1)*self.batch

                # 1.b Create batch x(k-1) + v*u(k-1)
                m_x, uk = self.get_parallel(x, self.batch, v)

                # 1.c Compute
                tmp_gaussian_loss = self.loss(self.model(m_x)).detach()                                 # Dim(bs)

                if verbose > 1:
                    print('\nINSIDE GRADIENT')
                    print('The Gaussian vector uk has shape:{}'.format(uk.shape))
                    print('The input x has shape:\t\t{}'.format(x.shape))
                    print('The input x + vu has shape:\t{}'.format(m_x.shape))

                # 1.d Compute Gradient
                fv = ((tmp_gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)            # Dim (bs, 1)
                G = fv * uk                                                                             # Dim (bs, channel*width*height)

                if verbose > 1:
                    print('Gaussian cycle loss has shape:\t{}'.format(tmp_gaussian_loss.shape))
                    print('Function approx has shape:\t{}'.format(fv.shape))
                    print('Gradient has shape:\t\t{}'.format(G.shape))

                G_tot[n] = torch.mean(G, axis=0).detach()

        return torch.mean(G_tot, axis=0).detach()



    """
    Compute the Inexact Condtion Gradient (Algorithm 3 of source article)
    """
    def compute_ICG(self, x, g, gamma, mu, verbose):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        g:              (torch.tensor)      The approximated gradient. Should be a 1D tensor
        gamma:          (float)             The update parameters of g
        mu:             (float)             The stopping criterion
        """
        # 1. Init variables
        y_old = x.view(-1).clone() # dim = (n_channel * width * height)
        t = 1
        k = 0
        # 2. Main cycle
        while(k==0):
            # 2.1 Compute gradient
            grad = g + gamma*(y_old - x.view(-1))

            # 2.2 Perform LMO
            # Infinity norm
            if self.L_type == -1:
                y_new = self.x_original.view(-1) - self.epsilon*torch.sign(grad)
            # L1 norm
            elif self.L_type == 1:
                raise NotImplementedError
            elif self.L_type == 2:
                y_new = self.x_original.view(-1) - (self.epsilon*grad)/torch.norm(grad, 2)
            # Generic Lp norm (1 < p < +inf)
            else:
                p = self.L_type
                gp = torch.abs(grad)**(1/p-1)
                h = torch.sign(grad) * (gp) / torch.norm(gp, p)
                y_new = self.x_original.view(-1) - self.epsilon*h

            # 2.3 Compute new function value
            h = torch.dot(grad, y_new - y_old)

            if verbose > 1:
                print('\nINSIDE ICG')
                print('Time t = {}'.format(t))
                print('The original gradient is:\n{}'.format(g))
                print('The ICG gradient is:\n{}'.format(grad))
                print('The new y is:\n {}'.format(y_new))
                print('The function h(y_new) is {}'.format(h))
                print('Mu is: {}'.format(mu))

            # 2.4 Check conditions
            if h >= -mu or t > self.max_t:
                k = 1
            else:
                y_old = ((t-1)/(t+1))*y_old + (2/(t+1))*y_new
                t += 1

        return self.project_boundaries(y_old.detach())


    """
    Check the boundaries of our constraint optimization problem
    """
    def project_boundaries(self, x):
        x[x > self.C[1]] = self.C[1]
        x[x < self.C[0]] = self.C[0]
        return x


"""
Accelerated Zero-order Stochastic Conditional Gradient (ZSCG) with Inexact Conditional Gradient (ICG) update.
Based on the paper:
    Zeroth-order Nonconvex Stochastic Optimization: Handling Constraints, High-Dimensionality and Saddle-Points∗
by:
    Krishnakumar Balasubramanian†1 and Saeed Ghadimi‡2
ALG.3
"""
class InexactAcceleratedZSCG(object):
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
    Perform an attack against the model given an input and the required run params
    """
    def run(self, x, v, n_gradient, alpha_k, gamma_k, mu_k, epsilon, L_type = -1, batch_size = -1, C = (0, 1),
            max_steps=100, verbose=0, additional_out=False, tqdm_disabled=False, max_t=1000):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        n_gradient:     (list)              Number of normal vector to generate at every step
        alpha_k:        (list)              Momentum at every step inside main cycle
        gamma_k:        (list)              Momentum at every step inside ICG
        mu_k:           (list)              Stoppinc criterion at every step k inside ICG
        max_t:          (int)               The maximum number of iteration inside of ICG.
        epsilon:        (float)             The upper bound of norm
        L_type:         (int)               Either -1 for L_infinity or x for Lx. Default is -1
        batch_size:     (int)               Maximum parallelization during the gradient estimation. Default is -1 (=n_grad)
        C:              (tuple)             The boundaires of the pixel. Default is (0, 1)
        max_steps:      (int)               The maximum number of steps. Default is 100
        verbose:        (int)               Display information or not. Default is 0
        additional_out: (bool)              Return also all the x. Default is False
        tqdm_disabled:  (bool)              Disable the tqdm bar. Default is False
        """

        x = x.to(self.device)

        # 1. Init class attributes
        self.x_original = x.clone()
        self.dim = x.shape
        self.total_dim = torch.prod(torch.tensor(x.shape))
        self.epsilon = epsilon
        self.L_type = L_type
        self.C = C
        self.batch = batch_size
        self.max_t = max_t

        # 2. Init list of results
        losses, outs = [], []
        input_list = []

        # 3. Main optimization cycle
        for ep in tqdm(range(max_steps), disable=tqdm_disabled):

            if verbose:
                print("---------------")
                print("Step number: {}".format(ep))

            # 3.1 Call the step
            x, gk = self.step(x, v, alpha_k[ep], gamma_k[ep], mu_k[ep], n_gradient[ep], verbose)
            x = x.reshape(self.dim[0], self.dim[1], self.dim[2]).detach()

            # 3.2 Compute loss
            out = self.model(x.view(1, self.dim[0], self.dim[1], self.dim[2]))
            loss = self.loss(out).view(-1, 1)

            # 3.3 Save results
            losses.append(loss.detach().cpu().item())
            outs.append(out.detach().cpu()[0, self.loss.neuron].item())
            if additional_out:
                input_list.append(x.cpu())

            # 3.4 Display current info
            if verbose:
                print("Loss:        {}".format(losses[-1]))
                print("Output:      {}".format(outs[-1]))

            # 3.5 Check Stopping criterion
            condition1 = (int(torch.argmax(out)) != self.loss.neuron) and (self.loss.maximise == 0)
            condition2 = (int(torch.argmax(out)) == self.loss.neuron) and (self.loss.maximise == 1)
            if condition1 or condition2:
                break

        if additional_out:
            return x, losses, outs, input_list
        return  x, losses, outs

    """
    Do an optimization step
    """
    def step(self, x, v, alpha, gamma, mu, mk, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        alpha:          (float)             Momentum
        gamma:          (float)             The update parameters of g
        mu:             (float)             The stopping criterion
        mk:             (int)               The number of Gaussian Random Vector to generate
        verbose:        (bool)              Display information or not. Default is 0
        """
        # Compute the approximated gradient
        g = self.compute_Gk(x, v, mk, verbose)
        # Call the inexact conditional gradient
        x_new = self.compute_ICG(x, g, gamma, mu, verbose).reshape(x.shape[0], x.shape[1], x.shape[2])
        x_new = (1-alpha)*x + alpha*x_new

        if verbose > 1:
            print("\nINSIDE STEP")
            print("Gradient has shape: {}".format(g.shape))
            print("Gradient is:\n{}".format(g))
            print("x_new has shape: {}".format(x_new.shape))
            print("x_new is:\n{}".format(x_new))

        return x_new.detach(), g.detach()


    """
    Prepare parallelization for the gradient estimation
    """
    def get_parallel(self, x, bs, v):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The current variable
        bs              (int)               The maximum bacth size
        v:              (int)               The Gaussian smoothing
        """
        uk     = torch.empty(bs, self.total_dim).normal_(mean=0, std=1).to(self.device) # Dim (bs, channel*width*height)
        img_u  = uk.reshape(bs, self.dim[0], self.dim[1], self.dim[2])                  # Dim (bs, channel, width, height)
        img_x  = x.expand(bs, self.dim[0], self.dim[1], self.dim[2])                    # Dim (bs, channel, width, height)
        m_x    = (img_x + v*img_u)

        return m_x, uk

    """
    Compute the Gv(x(k-1), chi(k-1), u(k)) in order to compute an approximation of the gradient of f(x(k-1), chi(k-1))
    """
    def compute_Gk(self, x, v, mk, verbose=0):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        v:              (float)             The gaussian smoothing
        mk:             (int)               The number of Gaussian Random Vector to generate
        verbose:        (bool)              Display information or not. Default is 0
        """

        # 1. Get objective functions
        # CASE BATCH_SIZE == N_GRADIENT
        if self.batch == -1:

            # 1.a Compute standard Loss
            standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))

            # 1.b Compute gaussian loss
            m_x, uk = self.get_parallel(x, mk, v)
            gaussian_loss = self.loss(self.model(m_x))

            if verbose > 1:
                print('\nINSIDE GRADIENT')
                print('The Gaussian vector uk has shape:{}'.format(uk.shape))
                print('The input x has shape:\t\t{}'.format(x.shape))
                print('The input x + vu has shape:\t{}'.format(m_x.shape))

            # 1.c Compute Gv(x(k-1), chi(k-1), u(k))
            fv = ((gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)        # Dim (mk, 1)
            G = fv * uk                                                                     # Dim (mk, channel*width*height)

            return torch.mean(G, axis=0).detach()

        # CASE BATCH_SIZE < N_GRADIENT
        else:

            # 1.a Compute standard loss
            standard_loss = self.loss(self.model(x.view(1, *list(self.dim))))                   # Dim (1)
            G_tot = torch.zeros(mk//self.batch, self.total_dim).to(self.device)                 # Dim (n_batches, hannel*width*height)

            #1.b Compute Gradient
            for n in range(mk//self.batch):
                from_, to_ = n*self.batch, (n+1)*self.batch

                # 1.b Create batch x(k-1) + v*u(k-1)
                m_x, uk = self.get_parallel(x, self.batch, v)

                # 1.c Compute
                tmp_gaussian_loss = self.loss(self.model(m_x)).detach()                                 # Dim(bs)

                if verbose > 1:
                    print('\nINSIDE GRADIENT')
                    print('The Gaussian vector uk has shape:{}'.format(uk.shape))
                    print('The input x has shape:\t\t{}'.format(x.shape))
                    print('The input x + vu has shape:\t{}'.format(m_x.shape))

                # 1.d Compute Gradient
                fv = ((tmp_gaussian_loss - standard_loss.expand(uk.shape[0]))/v).view(-1, 1)            # Dim (bs, 1)
                G = fv * uk                                                                             # Dim (bs, channel*width*height)

                if verbose > 1:
                    print('Gaussian cycle loss has shape:\t{}'.format(tmp_gaussian_loss.shape))
                    print('Function approx has shape:\t{}'.format(fv.shape))
                    print('Gradient has shape:\t\t{}'.format(G.shape))

                G_tot[n] = torch.mean(G, axis=0).detach()

        return torch.mean(G_tot, axis=0).detach()



    """
    Compute the Inexact Condtion Gradient (Algorithm 3 of source article)
    """
    def compute_ICG(self, x, g, gamma, mu, verbose):
        """
        Args:
        Name            Type                Description
        x:              (torch.tensor)      The variable of our optimization problem. Should be a 3D tensor (img)
        g:              (torch.tensor)      The approximated gradient. Should be a 1D tensor
        gamma:          (float)             The update parameters of g
        mu:             (float)             The stopping criterion
        """
        # 1. Init variables
        y_old = x.view(-1).clone() # dim = (n_channel * width * height)
        t = 1
        k = 0
        # 2. Main cycle
        while(k==0):
            # 2.1 Compute gradient
            grad = g + gamma*(y_old - x.view(-1))

            # 2.2 Perform LMO
            # Infinity norm
            if self.L_type == -1:
                y_new = self.x_original.view(-1) - self.epsilon*torch.sign(g)
            # L1 norm
            elif self.L_type == 1:
                raise NotImplementedError
            elif self.L_type == 2:
                y_new = self.x_original.view(-1) - (self.epsilon*g)/torch.norm(g, 2)
            # Generic Lp norm (1 < p < +inf)
            else:
                p = self.L_type
                gp = torch.abs(g)**(1/p-1)
                h = torch.sign(g) * (gp) / torch.norm(gp, p)
                y_new = self.x_original.view(-1) - self.epsilon*h

            # 2.3 Compute new function value
            h = torch.dot(grad, y_new - y_old)

            if verbose > 1:
                print('\nINSIDE ICG')
                print('Time t = {}'.format(t))
                print('The original gradient is:\n{}'.format(g))
                print('The ICG gradient is:\n{}'.format(grad))
                print('The new y is:\n {}'.format(y_new))
                print('The function h(y_new) is {}'.format(h))
                print('Mu is: {}'.format(mu))

            # 2.4 Check conditions
            if h >= -mu or t > self.max_t:
                k = 1
            else:
                y_old = (t-1)/(t+1)*y_old + 2/(t+1)*y_new
                t += 1

        return self.project_boundaries(y_old.detach())


    """
    Check the boundaries of our constraint optimization problem
    """
    def project_boundaries(self, x):
        x[x > self.C[1]] = self.C[1]
        x[x < self.C[0]] = self.C[0]
        return x



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
            x = nn.ReLU()(self.conv(x))
            return nn.Sigmoid()(self.linear(x.view(x.shape[0], -1)))

    epoch = 50
    m = [50]*epoch
    a = [0.9]*epoch
    v = 1


    net = Net()
    loss = SpecificSoftmaxMSE(neuron=2, y_true=0, dim=1)
    optim = ZeroSGD(model=net, loss=loss)

    x = torch.tensor([1])
    x, loss_curve, out, xs = optim.run(x.view(1, 1, 1), v, m, a, epsilon=0.5,
                                max_steps=epoch, stop_criterion = 0,
                                max_aux_step = 100, verbose=0, additional_out=True)

    min_, max_ = min(xs), max(xs)
    losses = []
    for i in tqdm(range(int(min_-1)*10, int(max_+1)*10)):
        x = torch.tensor([i/10]).to(torch.device('cuda'))
        out = net(x.view(1, 1, 1, 1))
        losses.append(loss(out))

    plt.plot([i/10 for i in range(int(min_-1)*10, int(max_+1)*10)], losses, label='Loss curve')
    plt.scatter(xs, loss_curve, label='Parameters')
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Loss')
    plt.title('Loss function')
    plt.grid()
    plt.show()
