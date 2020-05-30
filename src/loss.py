import torch
from torch import nn

"""
Abstract object for the Costum Loss. Child of nn.Module
"""
class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()


"""
Given a target neuron and a target (y_true).
Compute the Mean Squared Difference between the softmax output and the target
"""
class SpecificSoftmaxMSE(CustomLoss):

    def __init__(self, neuron, y_true=0, dim=1):
        """
        Args:
        Name     Type    Desc
        neuron:  int     The output neuron to minimize
        y_true   float   The desired activation
        dim      int     The softmax axis. Default is one for tensor with shape (n_batches, n_classes)
        """
        super().__init__()
        self.neuron = neuron
        self.y_true = y_true
        self.logits = nn.Softmax(dim=dim)

    """
    Compute the MSE after computing the softmax of input.
    Forward is implemented in the __call__ method of super
    """
    def forward(self, y_pred):
        """
        Args
        y_pred  torch.tensor The output of the networ. Preferable shape (n_batch, n_classes)
        """
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(1, -1)
        return 0.5*(self.y_true - self.logits(y_pred)[:, self.neuron])**2
