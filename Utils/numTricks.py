import numpy as np
from numba import jit


@jit(nopython=True)
def isClose(a, b, rtol=1e-05, atol=1e-08,):
    result = np.absolute(a - b) <= (atol + rtol * np.absolute(b))
    return result


@jit(nopython=True)
def isInt(a):
    result = isClose(np.ceil(a), a) or isClose(np.floor(a), a)
    return result


@jit(nopython=True)
def isZero(a):
    result = isClose(a, 0.)
    return result


# Function to decide whether a given matrix 'a' is singular or not. From
# https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


# reject outliers
# from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
# modified to use 3 stdev
def reject_outliers(data, m=3):
    array_data = np.array(data)
    return array_data[abs(array_data - np.mean(array_data)) < m * np.std(array_data)]


# Find nearest value in numpy array
# from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# ======================EOF
