import torch
from torch import nn

"""
Abstract object for the Custom Loss. Child of nn.Module
"""
class Loss(object):

    def __init__(self, neuron, maximise=0, is_softmax=False):
        """
        Args:
            Name       Type    Desc
            neuron     int     The output neuron to minimize
            maximise   bool    The desired activation
        """
        self.neuron = neuron
        self.maximise = maximise
        self.is_softmax = is_softmax

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


"""
Given a target neuron and a target (y_true).
Compute the Mean Squared Difference between the softmax output and the target
"""
class MSELoss(Loss):

    def __init__(self, neuron, maximise=0, is_softmax=False, dim=1):
        """
        Args:
            Name         Type    Desc
            neuron:      int     The output neuron to minimize
            maximise:    bool    The desired activation (0/1)
            is_softmax:  bool    Bool indicating if the model output is probability distribution. Default is False
            dim:         int     Dimension of softmax application. Default is 1
        """
        super().__init__(neuron, maximise, is_softmax)
        self.dim = dim

    """
    Compute the MSE after computing the softmax of input.
    Forward is implemented in the __call__ method of super
    """
    def forward(self, y_pred):
        """
        Args
            y_pred  torch.tensor The output of the network. Preferable shape (n_batch, n_classes)
        """
        # Deal with 1D input
        if len(y_pred.shape) == 1:
            y_pred = y_pred.view(-1, 1)
        # Case model output does not have non-softmax model output
        if not self.is_softmax:
            # Compute logits
            y_pred = nn.Softmax(dim=self.dim)(y_pred)
        # Return loss
        return 0.5*(int(self.maximise) - y_pred[:, self.neuron])**2



"""
Compute loss as defined in: "ZOO: Zeroth Order Optimization Based Black-box
Attacks to Deep Neural Networks without Training Substitute Models." [Chen et al]
"""
class ZooLoss(Loss):

    def __init__(self, neuron, maximise, transf=0, is_softmax=False, dim=1):
        """
        Args         Type      Desc
        neuron       int       If maximize is True is the desired output, the original class label otherwise
        maximize     bool      If True the attack is targeted, untargeted otherwise
        transf       float     Transferability parameter
        is_softmax:  bool      Bool indicating if the model output is probability distribution. Default is False
        dim:         int       Dimension of softmax application. Default is 1
        """
        super().__init__(neuron, maximise, is_softmax)
        self.transf = transf
        self.dim = dim


    def forward(self, conf):
        """
        Args    Type            Desc
        conf    torch_tensor    Matrix of size (n_batch, n_classes) containing the confidence score (output of the model)
        """
        # Deal with 1D input
        if len(conf.shape) == 1:
            conf = conf.view(-1, 1)

        # Avoid loss to be 0 in case of equal probability
        if self.maximise:
            conf[:, self.neuron] -= 1e-10
        else:
            conf[:, self.neuron] += 1e-10

        # Deal with non-softmax model output
        if not self.is_softmax:
            conf = nn.Softmax(dim=self.dim)(conf)

        # Compute log and neg_log matrices
        conf_log = torch.log(conf)
        conf_log_neg = torch.cat((conf_log[:, :self.neuron], conf_log[:, self.neuron+1:]), axis=1)

        # Compute Loss
        if self.maximise:
            # Targeted
            cln = torch.max(conf_log_neg, axis=1).values - conf_log[:, self.neuron]
        else:
            # Untargeted
            cln = conf_log[:, self.neuron] - torch.max(conf_log_neg, axis=1).values
        # Return loss
        return torch.max(cln, torch.zeros_like(cln)-self.transf)


# Unit testing
if __name__ == '__main__':
    mine = (1.999999, 2.2, 2.2, 2.2)
    mine = torch.tensor(mine).view(1, -1)
    print(mine)
    adv_loss = ZooLoss(0, maximise=1)
    print(adv_loss.forward(mine))
