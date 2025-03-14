# =======================================================================
# Averaging Correlations
#
# based on the explanations by Jan Seifert:
# https://medium.com/@jan.seifert/averaging-correlations-part-i-3adab6995042
# https://medium.com/@jan.seifert/averaging-correlations-part-ii-9143b546860b
#
# and the papers:
# Fisher, R. A. (1921). On the ’probable error’ of a coefficient of correlation
# deduced from a small sample. Metron, 1, 1–32.
# Retrieved from https://digital.library.adelaide.edu.au/dspace/bitstream/2440/15169/1/14.pdf
#
# Olkin, I., & Pratt, J. (1958). Unbiased estimation of certain correlation
# coefficients. The Annals of Mathematical Statistics, 29.
# https://doi.org/10.1214/aoms/1177706717
# =======================================================================
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, gammaln


class base_tr:
    def tr(self, r):
        return r

    def inv(self, r):
        return r


# Fisher z transformation or its inverse on a vector of correlation coefficients.
class fisher_z(base_tr):
    # z = 1/2 ln((1+r)/(1-r)) = artanh(r)
    def tr(self, r):
        return np.arctanh(r)

    # Fisher z inverse transformation
    # r = (exp(2z)-1)/(exp(2z)+1) = tanh(z)
    def inv(self, r):
        return np.tanh(r)


# MinVarZ
# FisherZ Correct sampple r with the equation by Olkin & Pratt (1958).
# Olkin, I. & Pratt, J.W. (1958). Unbiased Estimation of Certain
# Correlation Coefficients. The Annals of Mathematical Statistics,  29 (1),
# p. 201-211
# n here is the number of samples used to compute the original correlations (the timepoints?)
class Olkin_Pratt_approximate(base_tr):
    def __init__(self, n, k=(-7+9*np.sqrt(2))/2):
        if n < 5:
            raise ValueError("Sample size must be greater than 4")
        self.n = n
        self.k = k

    def tr(self, r):
        df = self.n - 1  # if all µ and σ are unkonwn (if µ was known it'd be n)
        G = r * (1 + (1 - r**2) / (2 * (df - self.k)))  # formula 2.7
        return G


# Olkin & Pratt (1958) formula to average correlations
# FisherZ Correct sample r with the equation by Olkin & Pratt (1958).
# from MinVarZ.pr:
# https://medium.com/@jan.seifert/averaging-correlations-part-ii-9143b546860b
# https://github.com/SigurdJanson/AveragingCorrelations/blob/master/CorrAggBias.R
# n here is the number of samples used to compute the original correlations (the timepoints?)
class Olkin_Pratt_precise(base_tr):
    def __init__(self, n):
        if n < 5:
            raise ValueError("Sample size must be greater than 4")
        self.n = n

    def tr(self, r):
        def fc2(t, r, df):
            return (1 - t) ** ((df - 4) / 2) / np.sqrt(1 - t + t * r ** 2) / np.sqrt(t)  #seems fastest

        def fc2_integral(r, df):
            # quad from scipy.integrate performs numerical integration equivalent to R's integrate().
            result, _ = quad(fc2, 0, 1, args=(r, df))
            return result

        n = self.n
        df = n - 1  # # if all µ and σ are unknown (if µ was know it'd be n)
        dfh = df / 2
        # Use value for gamma(0.5) to save computation time - https://oeis.org/A002161
        GammaOfHalf = np.log(1.772453850905516027298167483341145182797549456122387128213807789852911284591032181374950656738544665)
        # use lgamma tp prevent overflow
        # scipy.special.gammaln computes the logarithm of the gamma function, equivalent to lgamma in R.
        Component1 = np.exp(gammaln(dfh - 0.5) - gammaln(dfh - 1) - GammaOfHalf)  # Compute Component1
        Component2 = fc2_integral(r, df)  # Compute Component2
        # Calculate G
        G = r * Component1 * Component2
        return G


transf = fisher_z()  # correction algorithm
def weighted_avg(rs, ws):  # values, weights
    accum_attrib = 0
    accum_weight = 0
    for r, w in zip(rs,ws):
        accum_attrib += transf.tr(r) * w
        accum_weight += w
    res = transf.inv(accum_attrib/accum_weight)
    return res


# =======================================================================
# =======================================================================
# =======================================================================EOF