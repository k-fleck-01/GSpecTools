
###############################################################################
###############################################################################
"""
    @file examples/gauss_spectrum.py
    @brief Example to show the deconvolution on a numerically generated
    spectrum from a Gaussian input. Energy units are in GeV.
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
    NBINS = 100
    energy_bins = np.linspace(0.1, 10.0, NBINS+1)
    energy = 0.5 * (energy_bins[1:] + energy_bins[:-1])
    
    MU = 5.0; SIGMA = 1.5
    values = gauss_spectrum(energy, MU, SIGMA)

    original_spectrum = besiege.spectrum.Spectrum(energy_bins,
                                                  values,
                                                  np.zeros(NBINS))

    ### Numerically generate an electron/positron spectrum using the
    ### Bethe-Heitler cross section. A noise factor of 0.2 is used.
    target = besiege.cross_sections.MATERIALS_TABLE["tungsten"]
    target.set_thickness(100.0e-4) ### Define a 100um thick tungsten target

    bhgc_wrapper = lambda x,y : besiege.cross_sections.BHGammaConversion(x,y,target)

    generated_spectrum = original_spectrum.convolve(bhgc_wrapper)

    ### Deconvolve the spectrum and report the 95% HPDI as the uncertainty
    deconvolved_spectrum = generated_spectrum.deconvolve(bhgc_wrapper)


    ### Visualising the results
    fig, axs = plt.subplots(1, 2, layout="constrained")

    ### Numerically generated lepton spectrum
    generated_spectrum.visualise(ax=axs[0],
                                 xlabel="Energy (GeV)", ylabel=r"$dN/dE$ (1/GeV)")

    ### Comparison of original and deconvolved spectra
    deconvolved_spectrum.visualise(ax=axs[1], xlabel="Energy (GeV)",
                                   ylabel=r"$dN/dE$",
                                   color="tab:blue",
                                   label="reconstructed")
    original_spectrum.visualise(ax=axs[1], color="tab:red", label="original")

    plt.show()
