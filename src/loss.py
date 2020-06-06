import torch
from torch import nn

"""
Abstract object for the Costum Loss. Child of nn.Module
"""
class Loss():

    def __init__(self, neuron, maximise=0):

        """
        TODO: neuron -> target
        """
        """
        Args:
        Name       Type    Desc
        neuron     int     The output neuron to minimize
        maximise   bool    The desired activation
        """
        self.neuron = neuron
        self.maximise = maximise

    def __call__(self,args):
        return self.forward(args)

    def forward(self, args):
        raise NotImplementedError


"""
Given a target neuron and a target (y_true).
Compute the Mean Squared Difference between the softmax output and the target
"""
class SpecificSoftmaxMSE(Loss):

    def __init__(self, neuron, maximise=0, dim=1):
        super().__init__(neuron, maximise)
        """
        Args:
        Name       Type    Desc
        neuron:    int     The output neuron to minimize
        maximise   bool.   The desired activation (0/1)
        """
        self.logits = nn.Softmax(dim=dim)


    """
    Compute the MSE after computing the softmax of input.
    Forward is implemented in the __call__ method of super
    """
    def forward(self, y_pred):
        """
        Args
        y_pred  torch.tensor The output of the network. Preferable shape (n_batch, n_classes)
        """
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(1, -1)
        return 0.5*(self.maximise - self.logits(y_pred)[:, self.neuron])**2



"""
Compute loss as defined in: "ZOO: Zeroth Order Optimization Based Black-box
Attacks to Deep Neural Networks without Training Substitute Models." [Chen et al]
"""
class ZooLoss(Loss):

    def __init__(self, neuron, maximise, transf=0):
        """
        Args:
        Name       Type      Desc
        neuron     int       If maximize is True is the desired output, the original class label otherwise
        maximize   bool      If True the attack is targeted, untargeted otherwise
        transf     float     Transferability parameter
        """
        super().__init__(neuron, maximise)
        self.transf = transf


    def forward(self, conf):
        """
        Args      Type             Desc
        conf:     torch_tensor     Vector of length=n_classes containing the confidence score (output of the model)
        """
        conf = conf.reshape(-1)
        conf_log = torch.log(conf)
        conf_log_neg = torch.cat((conf_log[:self.neuron], conf_log[self.neuron+1:]))
        if self.maximise:
            # Targeted
            return torch.max(torch.max(conf_log_neg) - conf_log[self.neuron], -self.transf)
        else:
            # Untargeted
            return torch.max(conf_log[self.neuron] - torch.max(conf_log_neg), -self.transf)
