###############################################################################
###############################################################################
""" @file test_functions.py
    @brief Tests for src/reconstruction/analysis_functions.py
"""
###############################################################################
###############################################################################
from pytest import approx
import numpy as np
from scipy.stats import chi2

from besiege.reconstruction.analysis_functions import *

def test_kronecker_delta():
    assert kroenecker_delta(-1.0, 1.0) == 0.0
    assert kroenecker_delta(1.0, 1.0) == 1.0

def test_generate_trapz_kernel():
    dims = 10
    xvals = np.linspace(0.0, 1.0, dims)
    
    def func(x, y): ### Unit step function
        if x < y:
            return 0.0
        else:
            return 1.0

    ### Define solution matrix
    dx = xvals[1] - xvals[0]
    solmat = np.ones((dims, dims)) * dx
    solmat = np.tril(solmat)
    for i in range(dims):
        solmat[i, i] *= 0.5
        solmat[i, dims-1] *= 0.5

    kernel = generate_trapz_kernel(func, xvals)
    assert (kernel == solmat).all()

def test_lagrangian():
    ndim = 2
    matK = np.eye(ndim)
    data_vec = np.ones(ndim)
    pr_mean = np.zeros(ndim)

    sol = 0.5 * ndim + ndim * np.log(2.0)
    l_test = lagrangian([1.0, 1.0], matK, data_vec, pr_mean)
    assert l_test == approx(sol)

def test_estimate_hp():
    ndim = 2
    matK = np.eye(ndim)
    data_vec = np.ones(ndim)
    pr_mean = np.zeros(ndim)

    estBeta, estTheta, _ = estimate_hparameters(matK, data_vec, pr_mean)
    gamma = 1.0/estBeta + 1.0/estTheta
    assert gamma == approx(1.0)

def test_calculate_hpdi():
    ndim = 2
    mean = np.ones(ndim)
    covar = np.eye(ndim)

    alpha = 0.5
    error = calculate_hpdi(mean, covar, alpha)
    err2 = np.dot(error, error)
    assert err2 == approx(0.25*chi2(ndim).ppf(1.0 - alpha))
