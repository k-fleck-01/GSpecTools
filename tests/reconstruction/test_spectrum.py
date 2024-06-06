###############################################################################
###############################################################################
""" @file tests/reconstruction/test_spectrum.py
    @brief Tests for src/reconstruction/spectrum.py
"""
###############################################################################
###############################################################################
import pytest
from pytest import approx
import numpy as np

from besiege.reconstruction.spectrum import *
###############################################################################
def test_spectrum_init():
    nbins = 50
    basic_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              np.ones(nbins),
                              np.zeros(nbins))

    assert basic_spectrum.nbins == nbins
    
    with pytest.raises(RuntimeError):
        Spectrum(np.linspace(0.0, 1.0, nbins+1),
                 np.ones(nbins+1),
                 np.zeros(nbins))

    with pytest.raises(RuntimeError):
        Spectrum(np.linspace(0.0, 1.0, nbins+1),
                 np.ones(nbins),
                 np.zeros(nbins+1))

def test_getters():
    nbins = 50
    basic_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              np.ones(nbins),
                              np.zeros(nbins))

    xvals = np.linspace(0.0, 1.0, nbins+1)
    xcentres = np.linspace(1.0/2.0/nbins, 1.0 - 1.0/2.0/nbins, nbins)
    assert basic_spectrum.get_nbins() == nbins
    assert basic_spectrum.get_bin_edges() == approx(xvals)
    assert basic_spectrum.get_bin_centres() == approx(xcentres)
    assert basic_spectrum.get_bin_contents() == approx(np.ones(nbins))
    assert basic_spectrum.get_bin_errors() == approx(np.zeros(nbins))

def test_visualise():
    nbins = 50
    basic_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              np.ones(nbins),
                              np.zeros(nbins))

    _, ax = plt.subplots()
    basic_spectrum.visualise(ax, show_errs=True,
                             xlabel="x", ylabel="y", color="tab:blue")

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"

def test_scale_to_match(capsys):
    nbins = 50
    basic_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              np.ones(nbins),
                              np.zeros(nbins))

    other_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              2.0 * np.ones(nbins),
                              np.zeros(nbins))

    basic_spectrum.scale_to_match(other_spectrum)
    alpha = np.mean(basic_spectrum.get_bin_contents())
    assert alpha == approx(2.0)

    ### Test size mismatch
    other_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+2),
                              2.0 * np.ones(nbins+1),
                              np.zeros(nbins+1))
    with pytest.raises(RuntimeError):
        basic_spectrum.scale_to_match(other_spectrum)

    ### Test empty spectrum
    basic_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              np.zeros(nbins),
                              np.zeros(nbins))
    
    other_spectrum = Spectrum(np.linspace(0.0, 1.0, nbins+1),
                              2.0 * np.ones(nbins),
                              np.zeros(nbins))

    basic_spectrum.scale_to_match(other_spectrum)
    captured = capsys.readouterr()
    warn_str = "Warning: spectrum does not have any non-zero values. Abort scaling.\n"
    assert captured.out == warn_str 
