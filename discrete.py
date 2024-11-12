from dataclasses import dataclass
from typing import Any, Callable

from numpy import floating

from collection import scavenging, Droplet
from distribution import Distribution
import numpy as np
import numpy.typing as npt

from marshall_palmer import MarshallPalmer

#The number assigned to each bin is such that integrating each bin yields the same value
# as integrating the same range of the original distribution.

#This function produces a finite log-spaced bin range from 1 nm to 1cm with 100 bins.
BINS = np.logspace(-9,-3,100,dtype=np.float64)
FACTOR = BINS[1]/BINS[0]
BIN_ENDS = FACTOR*BINS
BIN_WIDTHS = BIN_ENDS-BINS
GEOMETRIC_CENTRES = BINS*np.sqrt(FACTOR)
ARITHMETIC_CENTRES = (BINS + BIN_ENDS)/2.0

def get_index(diameter:float | npt.NDArray[np.float_])->int | npt.NDArray[np.int_]:
    return np.floor(np.log(diameter / BINS[0]) / np.log(FACTOR))

@dataclass
class Discrete(Distribution):
    bin_heights: npt.NDArray[np.float_]
    def integrate_number_concentration(self, lower_diameter: npt.NDArray[np.float_],
                                       upper_diameter: npt.NDArray[np.float_]) -> floating[Any]:
        lower_index = get_index(lower_diameter)
        upper_index = get_index(upper_diameter)
        if lower_index == upper_index:
            return (upper_diameter-lower_diameter)*self.bin_heights[lower_index]
        total = np.sum(self.bin_heights[lower_index+1:upper_index]*BIN_WIDTHS[lower_index+1:upper_index])
        total += (BIN_ENDS[lower_index]-lower_diameter)*self.bin_heights[lower_index]
        total += (BINS[upper_index]-upper_diameter)*self.bin_heights[upper_index]
        return total

    @classmethod
    def make_from_distribution(cls,distribution: Distribution):
        bin_numbers = distribution.integrate_number_concentration(BINS,BIN_ENDS)
        number = np.sum(bin_numbers)
        bin_heights = bin_numbers/BIN_WIDTHS
        return cls(distribution.density,number,bin_heights)

    def get_number_concentration(self, diameter: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        bin_indices = get_index(diameter)
        return np.array([self.bin_heights[int(bin_index)] for bin_index in bin_indices])

    def evolve_bins(self, precipitation: MarshallPalmer, collection:Callable[[Droplet,Droplet],float], delta_time:float):
        scavenging_coefficients = scavenging(GEOMETRIC_CENTRES,self.density,precipitation,collection)
        scavenging_coefficients[scavenging_coefficients < 0] = 0
        print(scavenging_coefficients)
        self.bin_heights *= np.exp(-scavenging_coefficients*delta_time)