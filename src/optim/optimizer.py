"""
Generic, abstract optimizer. Implements a generic constructor and declares (
without defining them) run and step method. This class should be inherited by
every custom optimizer developed hereby.
"""


# Dependencies
import torch


# Class definition
class Optimizer(object):

    # Constructor
    def __init__(self, model, loss, device=torch.device('cpu')):
        # Save inner attributes
        self.device = device
        self.loss = loss
        # Set model in evaulation mode (no gradient)
        # and move it to chosen device
        self.model = model.eval().to(self.device)

    # Abstract method run
    def run(self, x, verbose=False):
        """
        This method starts from an input image and optimizes it values such
        that they produce the desired target output, as specified by the
        loss function attribute.

        Args:
        x (torch.Tensor):   3d input tensor (height x width x channels)
        verbose (bool):     Wether to print out optimization process or not
        """
        raise NotImplementedError

    # Abstract method step
    def step(self, *args, **kwargs):
        """
        Makes a single step through the optimization cycle defined in run method

        Args:
        x (torch.Tensor):  3d input tensor (height x width x channels)
        verbose (bool):    Wether to print out optimization process or not
        """
        raise NotImplementedError
