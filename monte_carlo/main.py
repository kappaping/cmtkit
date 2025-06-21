## Main function

import numpy as np
import time
import joblib
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append("../lattice/")
import lattice as ltc
sys.path.append("../tightbinding/")
import tightbinding as tb
import monte_carlo




# ----------
# Initialization.
# ----------

# Lattice structure.
ltype = "sq"
Nbl = [16, 16, 1]
rs = ltc.ltcsites(ltype, Nbl)[0]
bc = 1
file_lattice = "../../data/lattice/square/16161_bc_1"
NB = ltc.ltcpairdist(ltype, rs, Nbl, bc, toread=True, filet=file_lattice)[0]

# Ising model Hamiltonian.
Js = [0.,-1. / 2.]
H = tb.tbham(Js, NB, 1).real

# ----------
# Monte Carlo setup.
# ----------

# Generate a list of temperature to perform simulation at.
Ts=np.linspace(1,3.6,num=14).tolist()
# Set the type of Monte Carlo sweep. The default type is "metropolis", and the other possible types include "Wolff".
sweep_type = "wolff"
# Choose whether to print the information of iterations or write them to files.
to_print_or_write = "write"


# Define the function of Monte Carlo: Defining it as a function of temperature, which is convenient for parallel computing.
def compute(T):
    """
    Perform the simulation at a given temperature.
    """

    T = round(T, 2)
    # If the information is to be written to file, give the name of the file.
    if to_print_or_write == "write":
        file_name = f"../../data/monte_carlo/runtime/runtime_info_{T}"
    else:
        file_name = ""
    # Create an instance of Monte Carlo simulation.
    mc = monte_carlo.monte_carlo(neighbors=NB, to_randomize=True)
    # Equilibrate the simulation.
    mc.equilibrate(T, H, sweep_type=sweep_type, to_print_or_write=to_print_or_write, file_name=file_name)
    # Sample the states after equilibration.
    mc.sampling(T, H, sweep_type=sweep_type, to_print_or_write=to_print_or_write, file_name=file_name)
    print(f"Finished the computation at T = {T}")
    # Return the temperature and the sample.
    return [T, mc.sample]

# Initialize the data list to be stored.
data = []

# CPU parallelization: Use multiprocessing to distribute the computation to different CPU cores.
with ProcessPoolExecutor(max_workers=None) as executor:
    data = list(executor.map(compute, Ts))

# Store the data to a file.
joblib.dump(data, "../../data/monte_carlo/data_ising_16x16")


