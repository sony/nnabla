import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


class SequentialConverter(object):
    """Convert a given graph accoring to `converters`

    Args:
        converters (`list` of Converters): List of converters (e.g., FixedPointWeightConverter and FixedPointActivationConverter).
    """

    def __init__(self, converters=[]):
        self.converters = converters

    def convert(self, vroot, entry_variables):
        """Convert a given graph.

        Convert a given graph using the `converters` in the order of the registeration, i.e., sequentially.

        Args:
            vroot (:obj:`Variable`): NNabla Variable
            entry_variables (:obj:`Variable`): Entry variable from which the conversion starts.
        """

        for converter in self.converters:
            vroot = converter.convert(vroot, entry_variables)
        return vroot
