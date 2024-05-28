###############################################################################
###############################################################################
""" @file src/reconstruction/cross_sections.py
    @brief Bremsstrahlung and gamma conversion cross sections for use with the
    convolution and deconvolution methods in the Spectrum class.
"""
###############################################################################
###############################################################################
### Collection of values for N_A*\rho / A for common converter materials.
### To get the material parameter as used in the functions below, multiply by
### the converter thickness in cm.

MATERIALS_TABLE = {"tungsten" : None,
                   "bismuth" : None,
                   "tantalum" : None,
                   "lead" : None,
                  }

### TODO: Fix gamma conversion cross section function to use correct scaling
def BHGammaConversion(mat_param: float, energy: float, omega: float) -> float:
    """
    Energy differential cross section for creation of electron/positron
    pair in the field of a nucleus. Uses Klein's ultrarelativistic
    approximation, valid within a few percent at 1 GeV.

    Parameters
    ----------
    mat_param : float
        Material parameter of the 
    energy : float
        Energy of outgoing electron/positron in GeV.
    omega : float
        Energy of incident photon in GeV.

    Returns
    -------
    float

    """
    me = 0.511e-3 # GeV
    if omega <= 2*me: return 0. 
    
    x = energy/omega
    if (x < me/omega or x > 1.): 
        return 0. 
    else:
        y = 1. - x
        dsdE = 1. - (4./3.)*x*y
        dsdE /= omega
    
    if dsdE < 0.0: 
        return 0.0
    else:
        return mat_param*dsdE

###############################################################################
### TODO: Fix bremsstrahlung cross section to use correct scaling
def BHBremsstrahlung(mat_param: float, omega: float, energy: float) -> float:
    """
    Energy differential cross section for creation of a photon
    in the field of a nucleus (bremsstrahlung). Uses Klein's ultrarelativistic
    approximation, valid within a few percent at 1 GeV.

    Parameters
    ----------
    mat_param : float
        Material parameter (density*thickness/radiation length) of converter
        material.
    omega : float
        Energy of emitted photon in GeV.
    energy : float
        Energy of incoming electron/positron in GeV.

    Returns
    -------
    float

    """
    me = 0.511e-3 # GeV
    
    if energy <= 0. : return 0.
    
    y = omega/energy
    if y > 1. :
        return 0.
    else:
        dsdk = 4./3. - 4.*y/3. + y*y
        dsdk /= omega
        return mat_param*dsdk
