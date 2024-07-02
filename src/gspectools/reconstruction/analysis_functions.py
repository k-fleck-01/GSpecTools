###############################################################################
###############################################################################
""" @file src/reconstruction/analysis_functions.py
    @brief Functions used to perform Bayesian deconvolution.
"""
###############################################################################
###############################################################################
from typing import Callable

import numpy as np
import numpy.typing as npt
import scipy.linalg as linalg
import scipy.optimize as optimize
from scipy.special import gammainccinv

arrayf64 = npt.NDArray[np.float64]
###############################################################################
def kroenecker_delta(x: float, y: float) -> float:
    """Kroenecker delta (or discrete Dirace delta) function, equals
    1 if x = y; 0 otherwise.
    """
    return 1.0 if x == y else 0.0


def generate_trapz_kernel(func: Callable, xvalues: arrayf64) -> arrayf64:
    r"""Discretise a Volterra integral kernel of the second kind using the
    trapezium rule and return as a square matrix. Assumes evenly spaced
    abscissae.
        $$ K(s, x) = \int_s^{x_max} dx\ f(s, x) \circ $$
    """
    ndim = xvalues.size
    binwidth = xvalues[1] - xvalues[0]
    kernel_mat = np.zeros((ndim, ndim))

    for i, xx in enumerate(xvalues):
        for j, yy in enumerate(xvalues):
            dij = (1.0 - 0.5 * kroenecker_delta(i, j)) * (
                1.0 - 0.5 * kroenecker_delta(ndim - 1, j)
            )
            kernel_mat[i, j] = dij * binwidth * func(xx, yy)

    return kernel_mat


def lagrangian(
    params: list[float], matK: arrayf64, data_vec: arrayf64, pr_mean: arrayf64
) -> float:
    r"""Minimising Lagrangian to estimate hyperparameters; defined as the the
    negative natural logarithm of the marginalised likelihood function,
    $$ L = -\ln\det(\beta K K^T + \theta I) + g^T(\beta KK^T + \theta I)g $$
    """
    ndim = data_vec.size
    matI = np.eye(ndim)  ### Identity matrix
    pBeta, pTheta = params  ### Hyperparameters

    ### Define B^-1 = \beta K K^T + \theta I
    matB = (1.0 / pTheta) * matK @ matK.transpose() + (1.0 / pBeta) * matI
    matB_inv = linalg.inv(matB)
    detB = linalg.det(matB)

    ### Include transform term for non-zero mean
    matL = pBeta * matK.transpose() @ matK + pTheta * matI
    matL_inv = linalg.inv(matL)
    vec0 = pBeta * pTheta * matB @ matL_inv @ pr_mean

    z = (data_vec - vec0).transpose() @ matB_inv @ (data_vec - vec0)
    return z + np.log(detB)


def estimate_hparameters(
    matK: arrayf64, data_vec: arrayf64, pr_mean: arrayf64
) -> tuple[float, float]:
    """Estimate the precision hyperparameters using the marginalised likelihood
    function by minimizing its negative natural logarithm.
    """
    ### Add constraints that the hyperparameters must be > 0
    bnds = optimize.Bounds(1.0e-6, np.inf)
    knorm = linalg.norm(matK, ord=np.inf)
    matK /= knorm  ### Divide matrix by its maximum norm for regularisation

    ### Estimating hyperparameters
    p_opt = optimize.minimize(
        lagrangian, (1.0, 1.0), args=(matK, data_vec, pr_mean), bounds=bnds
    )
    pBeta_est, pTheta_est = p_opt.x

    ### Rescale matrix and theta hyperparameter
    pTheta_est *= knorm * knorm
    matK *= knorm
    return (pBeta_est, pTheta_est)


def calculate_hpdi(mean: arrayf64, covar: arrayf64, alpha: float) -> arrayf64:
    """Calculate the highest posterior density interval (HPDI) for a multivariate
    normal distribution at the (1 - alpha) level.
    """
    ndim = mean.size
    ### Get critical radius
    crit_radius = np.sqrt( 2.0 * gammainccinv(ndim / 2.0, alpha) )
    rad_uvec = np.ones_like(mean) / np.sqrt(ndim)

    ### Calculate error
    matL = linalg.cholesky(covar, lower=True)
    error = crit_radius * matL @ rad_uvec
    return error
