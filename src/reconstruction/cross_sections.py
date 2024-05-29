###############################################################################
###############################################################################
""" @file src/reconstruction/cross_sections.py
    @brief Bremsstrahlung and gamma conversion cross sections for use with the
    convolution and deconvolution methods in the Spectrum class.
"""


###############################################################################
###############################################################################
class Material:
    """Simple class to hold the information needed to evaluate
    cross sections. Thickness in cm, radiation length in g/cm^2 and density in
    g/cm^3.
    """

    AVOGADRO = 6.022e23

    def __init__(
        self,
        atomic_A: float,
        rad_length: float,
        density: float,
        thickness: float | None = None,
    ) -> None:
        self.atomic_A = atomic_A
        self.rad_length = rad_length
        self.density = density
        self.thickness = thickness

    def set_thickness(self, thickness: float) -> None:
        """Manually set the thickness."""
        self.thickness = thickness

    @property
    def mat_coefficient(self) -> float:
        """Calculate the material coefficient
        $$ N_A \rho t / A $$
        """
        if self.thickness is not None:
            param = self.thickness * self.density / self.atomic_A
            return Material.AVOGADRO * param
        else:
            raise ValueError("Material thickness has not been specified.")

    @property
    def cross_section_factor(self) -> float:
        """Calculate the scaling factor for cross sections
        $$ A / X_0 / N_A $$.
        """
        return self.atomic_A / self.rad_length / Material.AVOGADRO


### Set of common materials for conversion targets, uses default thickness.
MATERIALS_TABLE = {
    "tungsten": Material(atomic_A=74.0, rad_length=6.76, density=19.30),
    "bismuth": Material(atomic_A=83.0, rad_length=6.29, density=9.747),
    "tantalum": Material(atomic_A=73.0, rad_length=6.82, density=16.65),
    "lead": Material(atomic_A=82.0, rad_length=6.37, density=11.35),
}


###############################################################################
### Cross section functions
###############################################################################
def BHGammaConversion(energy: float, omega: float, targ: Material) -> float:
    """Bethe-Heitler cross section for electron-positron production in the
    nuclear field in units of cm^2. Assumes the ultrarelativistic complete
    screening approximation.
    """

    me = 0.511e-3  # GeV
    if omega <= 2 * me:
        return 0.0

    x = energy / omega
    if x < me / omega or x > 1.0:
        return 0.0
    else:
        y = 1.0 - x
        dsdE = 1.0 - (4.0 / 3.0) * x * y
        dsdE /= omega

    if dsdE < 0.0:
        return 0.0
    else:
        return dsdE * targ.cross_section_factor


def BHBremsstrahlung(omega: float, energy: float, targ: Material) -> float:
    """Bethe-Heitler cross section for bremsstrahlung in the
    nuclear field in units of cm^2. Assumes the ultrarelativistic complete
    screening approximation.
    """
    me = 0.511e-3  # GeV

    if energy <= 0.0:
        return 0.0

    y = omega / energy
    if y > 1.0:
        return 0.0
    else:
        dsdk = 4.0 / 3.0 - 4.0 * y / 3.0 + y * y
        dsdk /= omega
        return dsdk * targ.cross_section_factor
