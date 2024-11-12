import numpy as np
from numpy import typing as npt
from distribution import Distribution
from scipy.integrate import quad

class MarshallPalmer(Distribution):
    size_constant: float

    def __init__(self, precipitation_rate: float):
        self.number = 8000e3
        self.density = 1000
        self.size_constant = 4100*np.power(precipitation_rate,-0.21)

    def integrate_number_concentration(self, lower_diameter: float | npt.NDArray[np.float_],
                                       upper_diameter: float | npt.NDArray[np.float_]) -> float:
        return quad(self.get_number_concentration, lower_diameter, upper_diameter)[0]


    def get_number_concentration(self, diameter: float) -> float:
        return self.number * np.exp(-self.size_constant*diameter)