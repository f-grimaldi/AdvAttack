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
