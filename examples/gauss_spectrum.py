
###############################################################################
###############################################################################
"""
    @file examples/gauss_spectrum.py
    @brief Example to show the deconvolution on a numerically generated
    spectrum from a Gaussian input. Energy units are in GeV. Final plots are
    scaled to their maximum values for better visualisation and comparison of
    spectral shape.
"""
###############################################################################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt

import besiege
###############################################################################
def gauss_spectrum(x, mu, sigma):
    """Defines a Gaussian spectrum of unit integral, with mean 'mu' and
    standard deviation'sigma'.
    """
    amp = 1.0 / np.sqrt(2.0 * np.pi) / sigma
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

###############################################################################
if __name__ == '__main__':
    ### Define the energy space and a Gaussian spectrum
    NBINS = 25
    energy_bins = np.linspace(0.1, 10.0, NBINS+1)
    energy = 0.5 * (energy_bins[1:] + energy_bins[:-1])
    
    MU = 5.0; SIGMA = 1.2
    values = gauss_spectrum(energy, MU, SIGMA)

    original_spectrum = besiege.spectrum.Spectrum(energy_bins,
                                                  values,
                                                  np.zeros(NBINS))

    ### Numerically generate an electron/positron spectrum using the
    ### Bethe-Heitler cross section. A noise factor of 0.2 is used.
    target = besiege.cross_sections.MATERIALS_TABLE["tungsten"]
    target.set_thickness(100.0e-4) ### Define a 100um thick tungsten target

    def convolution_kernel(x, y):
        csec = besiege.cross_sections.BHGammaConversion(x,y,target)
        return target.mat_coefficient * csec

    generated_spectrum = original_spectrum.convolve(convolution_kernel, fano=0.0)

    ### Deconvolve the spectrum and report the 95% HPDI as the uncertainty
    deconvolved_spectrum = generated_spectrum.deconvolve(convolution_kernel)

    ### Visualising the results
    fig, axs = plt.subplots(1, 2, figsize=(8,6), layout="constrained")

    ### Numerically generated lepton spectrum
    scale_factor = 1.0 / generated_spectrum.get_bin_contents().max()
    generated_spectrum.scale(scale_factor)
    generated_spectrum.visualise(ax=axs[0])

    ### Comparison of original and deconvolved spectra
    scale_factor = 1.0 / deconvolved_spectrum.get_bin_contents().max()
    deconvolved_spectrum.scale(scale_factor)
    deconvolved_spectrum.visualise(ax=axs[1], show_errs=True,
                                   color="tab:blue",
                                   label="reconstructed")

    scale_factor = 1.0 / original_spectrum.get_bin_contents().max()
    original_spectrum.scale(scale_factor)
    original_spectrum.visualise(ax=axs[1], color="tab:red", label="original")

    fig.supxlabel("Energy (GeV)")
    fig.supylabel(r"$dN/dE$ (norm. units)")
    axs[1].legend(loc="upper right")
    plt.show()
