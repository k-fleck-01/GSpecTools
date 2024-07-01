###############################################################################
###############################################################################
""" @file src/reconstruction/spectrum.py
    @brief Histogram-like Spectrum class which allows for visualisation and
    deconvolution of results.
"""
###############################################################################
###############################################################################
from typing import Self, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

from . import analysis_functions as af


###############################################################################
class Spectrum:
    """Histogram-like class to hold spectrum data in a binned structure."""

    def __init__(self, xbins: np.ndarray, yvals: np.ndarray, yerrs: np.ndarray) -> None:
        nedges = xbins.size
        nbins = nedges - 1
        if yvals.size != nbins or yerrs.size != nbins:
            raise RuntimeError(
                "Number of bins does not match number of specified values or errors."
            )

        self.nbins = nbins
        self.bin_edges = xbins
        self.bin_centres = 0.5 * (xbins[:-1] + xbins[1:])
        self.bin_contents = yvals
        self.bin_errors = yerrs

    ### Some getters to avoid overwriting
    def get_nbins(self) -> int:
        return self.nbins

    def get_bin_edges(self) -> np.ndarray:
        return self.bin_edges

    def get_bin_centres(self) -> np.ndarray:
        return self.bin_centres

    def get_bin_contents(self) -> np.ndarray:
        return self.bin_contents

    def get_bin_errors(self) -> np.ndarray:
        return self.bin_errors

    def visualise(
        self,
        ax: plt.Axes,
        show_errs: bool = False,
        xlabel: str = "", ylabel: str = "",
        **kwargs
    ) -> None:
        """Plots the spectrum onto a specified Axes object. Shows the
        associated error if enabled (default off).
        """
        xbins = self.bin_edges
        xcentres = self.bin_centres
        yvals = self.bin_contents
        yerrs = self.bin_errors

        patches = ax.stairs(yvals, xbins, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Show error region
        if show_errs:
            lower = yvals - yerrs
            lower[lower < 0.0] = 0.0

            upper = yvals + yerrs
            ax.fill_between(xcentres, y1=lower, y2=upper,
                            alpha=0.25, color=patches.get_edgecolor())

    def scale(self, other: float | Self) -> None:
        """Scale the contents of this spectrum. If 'other' is a float, then
        contents are scaled by a constant factor. If 'other' is a Spectrum,
        then contents are scaled to match the contents of the other spectrum
        using a linear least squares method.

            alpha_opt = <y_true, y_data>/<y_data, y_data>

        Parameters
        ----------
        other : float | Spectrum
            Float or Spectrum to scale by.
        """
        if isinstance(other, float):
            self.bin_contents *= other
            self.bin_errors *= other

        elif isinstance(other, Spectrum):
            y_true = other.get_bin_contents()
            y_data = self.get_bin_contents()

            if y_data.size != y_true.size:
                raise RuntimeError(
                    "Spectrum does not have the same number of bins as the other spectrum."
                )

            if (y_data == 0.0).all():
                print("Warning: spectrum does not have any non-zero values. Abort scaling.")
                return

            alpha = np.dot(y_true, y_data) / np.dot(y_data, y_data)

            self.bin_contents *= alpha
            self.bin_errors *= alpha

    def convolve(self, func: Callable, fano: Optional[float] = None) -> "Spectrum":
        r"""Convolution of the Spectrum object with some function. Mathematically, this
        calculates
            $$ \int_s^{x_max} dx\ K(s, x) * f(x) $$
        where $K(s,x)$ is the kernel function specified by 'func' and $f(x)$ is
        represented by the Spectrum object.
        """
        # Get kernel matrix
        matK = af.generate_trapz_kernel(func, self.bin_centres)
        conv_yy = matK @ self.bin_contents
        conv_errs = matK @ self.bin_errors

        if fano is not None:
            conv_yy = np.random.normal(conv_yy, fano * np.sqrt(conv_yy))

        convSp = Spectrum(self.bin_edges.copy(), conv_yy, conv_errs)
        return convSp

    def _volterra_solve(self, func: Callable) -> np.ndarray:
        """Solves the Volterra integral equation of the second kind using
        a back-stepper solution. Adapted from Numerical Recipes by W. Press.
        """
        xx = self.bin_centres
        yy = self.bin_contents
        nn = xx.size
        vsol = np.zeros(nn)

        hh = xx[1] - xx[0]

        vsol[-1] = yy[-1] / func(xx[-1], xx[-1]) / hh
        for i in range(nn - 1, -1, -1):
            vsol[i] = -yy[i] + 0.5 * func(xx[i], xx[-1]) * vsol[-1] * hh
            for j in range(i + 1, nn - 2):
                vsol[i] += func(xx[i], xx[j]) * vsol[j] * hh
            vsol[i] /= -0.5 * func(xx[i], xx[i]) * hh

        return vsol

    def deconvolve(self, func: Callable, alpha: float = 0.05) -> "Spectrum":
        """Solves the Volterra integral equation of the second kind using
        Bayesian deconvolution. Returns the maximum a posteriori (MAP) estimate
        as the solution and the specified HPDI (95%, by default) as the
        error.
        """
        xx = self.bin_centres
        binwidths = self.bin_edges[1:] - self.bin_edges[:-1]
        yy = self.bin_contents / binwidths
        ndim = yy.size

        # Get kernel matrix
        matK = af.generate_trapz_kernel(func, xx)

        # Get prior mean using Volterra solution
        pr_mean = self._volterra_solve(func)

        # Estimate hyperparameters and calculate posterior mean and covariance
        # Iterate until convergence
        ITERMAX = 10
        TOLERANCE = 1.0e-10

        # Defaults in case iteration fails
        mean = pr_mean
        covar = np.diag(np.sqrt(pr_mean))

        for i in range(ITERMAX):
            pBeta_est, pTheta_est = af.estimate_hparameters(matK, yy, pr_mean)

            # Calculate MAP (mean) and covariance
            matA = pBeta_est * matK.transpose() @ matK + pTheta_est * np.eye(ndim)
            covar = np.linalg.inv(matA)

            mean = pBeta_est * matK.transpose() @ yy + pTheta_est * pr_mean
            mean = np.abs(covar @ mean)

            if np.linalg.norm(mean - pr_mean, ord=2) < TOLERANCE:
                print(f"Convergence reached after {i} iterations.")
                break
            else:
                pr_mean = mean

        # Get HPD interval
        error = af.calculate_hpdi(mean, covar, alpha)
        error[-1] = 0.0 ### Avoid numerical overestimation of error in last bin
        dSpectrum = Spectrum(self.bin_edges.copy(), mean, np.abs(error))
        return dSpectrum
