import matplotlib.pyplot as plt
import numpy as np

from collection import slinn, Droplet
from discrete import GEOMETRIC_CENTRES, Discrete, BINS, BIN_ENDS
from distribution import DistributionSum
from lognormal import LogNormal
from marshall_palmer import MarshallPalmer

if __name__ == "__main__":
    precipitation = MarshallPalmer(1)

    rc1 = LogNormal(1000,3200e6,0.02e-6,0.161)
    rc2 = LogNormal(1000,2900e6,0.116e-6,0.217)
    rc3 = LogNormal(1000,0.3e6,1.8e-6,0.380)
    remote_continental = DistributionSum([rc1,rc2,rc3])

    plt.plot(GEOMETRIC_CENTRES,remote_continental.get_volume_concentration(GEOMETRIC_CENTRES)*GEOMETRIC_CENTRES,label="Log Normal")
    plt.xscale("log")
    plt.xlabel("Aerosol Diameter / m")
    plt.ylabel("dV/ddp / m3 m-3")
    discrete = Discrete.make_from_distribution(remote_continental)


    plt.step(BIN_ENDS, discrete.get_volume_concentration(GEOMETRIC_CENTRES)*GEOMETRIC_CENTRES,label="Discrete")
    discrete.evolve_bins(precipitation,slinn,1800)
    plt.step(BIN_ENDS, discrete.get_volume_concentration(GEOMETRIC_CENTRES)*GEOMETRIC_CENTRES, label="Discrete after 0.5 h at 1 mm/hr")
    discrete.evolve_bins(precipitation, slinn, 1800)
    plt.step(BIN_ENDS, discrete.get_volume_concentration(GEOMETRIC_CENTRES) * GEOMETRIC_CENTRES,
             label="Discrete after 1 h at 1 mm/hr")
    discrete.evolve_bins(precipitation, slinn, 3600*9)
    plt.step(BIN_ENDS, discrete.get_volume_concentration(GEOMETRIC_CENTRES) * GEOMETRIC_CENTRES,
             label="Discrete after 10 h at 1 mm/hr")
    plt.legend(loc="upper left")
    plt.show()

    droplet_sizes = [0.1e-3,0.5e-3,1e-3,4e-3]
    precipitations = [MarshallPalmer(0.1),MarshallPalmer(1),MarshallPalmer(10),MarshallPalmer(100)]
    aerosols = np.array([Droplet(diameter,1000) for diameter in GEOMETRIC_CENTRES])

    for droplet_size in droplet_sizes:
        droplet = Droplet(droplet_size,1000)
        y_values = [slinn(aerosol,droplet) for aerosol in aerosols]
        plt.plot(GEOMETRIC_CENTRES,y_values,label=f"{droplet_size:.2} m")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Aerosol Diameter / m")
    plt.ylabel("Collection Efficiency")
    plt.show()