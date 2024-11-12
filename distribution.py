from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

@dataclass
class Distribution(ABC):
    density: float
    number: float
    @abstractmethod
    def get_number_concentration(self, diameter: float) -> float:
        pass
    def get_area_concentration(self,diameter: float) -> float:
        return np.pi*diameter**2*self.get_number_concentration(diameter)
    def get_volume_concentration(self,diameter: float) -> float:
        return np.pi/6.0*diameter**3*self.get_number_concentration(diameter)
    def get_mass_concentration(self,diameter: float) -> float:
        return self.density*self.get_volume_concentration(diameter)
    @abstractmethod
    def integrate_number_concentration(self,lower_diameter: float|npt.NDArray[np.float_], upper_diameter:float|npt.NDArray[np.float_]) -> float:
        pass

class DistributionSum(Distribution):
    distributions: List[Distribution]

    def __init__(self, distributions: List[Distribution]):
        self.distributions = distributions
        self.density = distributions[0].density
        self.number = np.sum([distribution.number for distribution in distributions])

    def get_number_concentration(self, diameter: float) -> float:
        return np.sum([distribution.get_number_concentration(diameter) for distribution in self.distributions],axis=0)

    def integrate_number_concentration(self, lower_diameter: float | npt.NDArray[np.float_],
                                       upper_diameter: float | npt.NDArray[np.float_]) -> float:
        return np.sum([distribution.integrate_number_concentration(lower_diameter,upper_diameter) for distribution in self.distributions],axis=0)