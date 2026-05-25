# Copyright (c) 2015 Jonas Rauber
# License: The MIT License (MIT)
# See: https://github.com/jonasrauber/randn-matlab-python
#
# Usage:
# For the original Numpy randn implementation, use:
# from numpy.random import randn as randn
#
# For the Matlab-identical implementation, use:
# from Utils.randn2 import randn2 as randn


from numpy import sqrt
from numpy.random import rand
from scipy.special import erfinv

def randn2(*args,**kwargs):
    '''
    Calls rand and applies inverse transform sampling to the output.
    '''
    uniform = rand(*args, **kwargs)
    seq = sqrt(2) * erfinv(2 * uniform - 1)
    return seq.flatten().reshape(args, order="F")

