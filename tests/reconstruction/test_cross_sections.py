###############################################################################
###############################################################################
""" @file test_cross_sections.py
    @brief Tests for src/reconstruction/cross_sections.py
"""
###############################################################################
###############################################################################
import pytest
from pytest import approx

from besiege.reconstruction.cross_sections import *
###############################################################################
def test_material():
    ### Testing default material
    material = Material(atomic_A=1.0,
                        rad_length=1.0,
                        density=1.0)

    assert material.thickness is None
    with pytest.raises(ValueError):
        material.mat_coefficient

    material.set_thickness(1.0)
    assert material.thickness == approx(1.0)

    mat_param = material.mat_coefficient
    assert mat_param == approx(6.022e23)

    cs_factor = material.cross_section_factor
    assert cs_factor == approx(0.16606e-23)

def test_gamma_conversion():
    target = Material(1.0, 1.0, 1.0, 1.0)
    scale = target.cross_section_factor

    ### Photon energy less than threshold
    assert BHGammaConversion(0.5e-4, 1.0e-3, target) == approx(0.0)

    ### Electron energy greater than photon energy
    assert BHGammaConversion(1.0, 0.5, target) == approx(0.0)

    ### Electron energy = omega/2.0
    assert BHGammaConversion(0.5, 1.0, target)/scale == approx(2.0/3.0)

def test_bremsstrahlung():
    target = Material(1.0, 1.0, 1.0, 1.0)
    scale = target.cross_section_factor

    ### Electron energy less than threshold
    assert BHBremsstrahlung(0.1, -1.0, target) == approx(0.0)
    ### Photon energy greater than electron energy
    assert BHBremsstrahlung(1.0, 0.5, target) == approx(0.0)
    ### Photon energy = energy/2.0
    assert BHBremsstrahlung(0.5, 1.0, target)/scale == approx(11.0/6.0)
