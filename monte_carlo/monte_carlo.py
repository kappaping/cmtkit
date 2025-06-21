## Monte Carlo module

'''Monte Carlo module: Functions of Monte Carlo'''

import math
import numpy as np
import random

import sys
sys.path.append('../')
import lattice as ltc




class monte_carlo():
    # A class of Monte Carlo simulation.
    # ----------
    # Attributes:
    # type_flavor: str, indicates the type of the site flavors. Possible values include "Ising" and "Heisenberg".
    # state: numpy.ndarray, the latest state in the Monte Carlo simulation.
    # sample: list of numpy arrays, the sampled states after equilibration.
    # neighbors: numpy.ndarray, a NxN matrix of lattice size N.
    # ----------


    def __init__(self, type_flavor="Ising", state=np.array([]), sample=[], neighbors=np.array([]), to_randomize=False):
        """
        Constructor function: Construct the Monte Carlo.
        """

        # Assign the attributes.
        self.type_flavor = type_flavor
        self.state = state
        self.sample = sample
        self.neighbors = neighbors
        # Initialize the simulation by randomization.
        if to_randomize:
            self.randomize()


    def randomize(self):
        """
        Randomize the Monte Carlo state.
        """

        # Clean the sample to initialize the simulation.
        self.sample = []
        # Generate a random state based on the flavor type.
        if self.type_flavor == "Ising":
            # Generate a state of randomized Ising flavors 1 or -1.
            state = np.random.choice([-1, 1], size=(self.neighbors.shape[0], 1))
        elif self.type_flavor == "Heisenberg":
            # Generate a state of randomized Heisenberg flavors, which are 3-component normal vectors.
            state = np.random.uniform(-1, 1, size=(self.neighbors.shape[0], 3))
            for spin in state:
                spin = spin / np.linalg.norm(spin)
        self.state = state


    def sweep(self, T, H, sweep_type="metropolis"):
        """
        Perform a Monte Carlo sweep to update the state.
        """

        # Implement the Monte Carlo sweep based on the chosen sweep type.
        if sweep_type == "metropolis":
            self.metropolis(T, H)
        elif sweep_type == "wolff":
            self.wolff(T, H)


    def metropolis(self, T, H):
        """
        Metropolis sweep: Flip all sites in random orders by the metropolis method.
        """

        # Generate the random order of sites for flavor flipping.
        sites_sweep = list(range(self.neighbors.shape[0]))
        random.shuffle(sites_sweep)
        # Flip the flavors iteratively.
        for site in sites_sweep:
            # Get the latest state.
            state = self.state.copy()
            # Compute the energy of the state.
            E_old = np.sum(H * np.dot(state, state.T)) 
            # Flip the flavor of the chosen site.
            if self.type_flavor == "Ising":
                # Ising flip: Switch 1 and -1.
                state[site] = -1 * state[site]
            elif self.type_flavor == "Heisenberg":
                # Heisenberg flip: Flip to another random normal vector.
                spin = np.random.uniform(-1, 1, size=3)
                state[site] = spin / np.linalg.norm(spin)
            # Compute the energy of the flipped state.
            E_new = np.sum(H * np.dot(state, state.T))
            # Get the energy change under the flip.
            dE = E_new - E_old
            # Determine whether to accept the flip by a probability P = e^{-dE / T}.
            if dE < 0 or (dE > 0 and random.uniform(0., 1.) < math.e**(-dE / T)):
                self.state = state


    def wolff(self, T, H):
        """
        Wolff cluster update: Flip a cluster connected to a randomly chosen site.
        """
        
        # Initialize the cluster by choosing a random site.
        cluster = [random.choice(list(range(self.neighbors.shape[0])))]
        # Create a test list the same as the cluster. This list contains the sites whose neighbors need to be examined.
        list_test = cluster.copy()
        # Get the size of the lattice.
        size_lattice = H.shape[0]
        # Get the neighbors on the lattice.
        NB = self.neighbors
        # Get the latest state.
        state = self.state.copy()
        # Determine the cluster iteratively by examining the neighbors. Continue until the increasing test list is exhausted.
        while len(list_test) > 0:
            # Consider the site at the head of the test list.
            site_0 = list_test[0]
            # Get the neighbors of this site.
            NB_0 = NB[site_0]
            # Examine the neighbors of this site.
            for site_1 in range(size_lattice):
                # Check if the site is new to the cluster and is a neighbor of the test site.
                if site_1 not in cluster and NB_0[site_1] == 1:
                    # If the neighbor has the same flavor as the test site, accept it to the cluster and the test list with a probability 1 - e^{-2J/T}.
                    if np.dot(state[site_0], state[site_1]) == 1 and random.uniform(0., 1.) < 1 - math.e**((-2 * (-2 * H[site_0, site_1])) / T):
                        cluster.append(site_1)
                        list_test.append(site_1)
            # Remove the test site from the test list.
            list_test = list_test[1:]
        # Flip the whole cluster.
        for site in cluster:
            state[site] = -1 * state[site]
        # Update the state.
        self.state = state


    def equilibrate(self, T, H, sweep_type="metropolis", M_eq=10000, M_avg=100, error_max=1e-8, to_print_or_write="print", file_name=""):
        """
        Equilibrate the Monte Carlo simulation at a given temperature.
        """

        # Initialize the average energy and a list for its computation.
        E_avg = 1.
        Es = []
        # Get the lattice size.
        size_lattice = self.neighbors.shape[0]
        # Perform the Monte Carlo sweep many times to equilibrate the simulation.
        for m in range(M_eq):
            # Get the original state.
            state_old = self.state.copy()
            # Perform the Monte Carlo sweep.
            self.sweep(T, H, sweep_type=sweep_type)
            # Get the updated state.
            state = self.state.copy()
            # Get the energy.
            E = np.sum(H * np.dot(state, state.T))
            # Update the list of energies for the computation of average energy.
            Es = Es[m >= M_avg:] + [E]
            # Compute the average energy and its iterative error.
            E_avg_old = E_avg
            E_avg = sum(Es) / len(Es)
            E_avg_error = (E_avg - E_avg_old) / size_lattice
            # Get the number of sites that have been flipped.
            num_flip = size_lattice - (state.reshape(-1) - state_old.reshape(-1)).tolist().count(0)
            # Print the information of current iteration, either directly or in a file.
            info = f"Equilibration at T = {T} [{m}/{M_eq}]: Energy = {E_avg}, error = {E_avg_error}, number of flips = {num_flip}"
            if to_print_or_write == "print":
                print(info)
            elif to_print_or_write == "write":
                with open(file_name, "a") as file_info:
                    file_info.write(info + "\n")


    def sampling(self, T, H, sweep_type="metropolis", M_sample=100, M_sample_diff=100, to_print_or_write="print", file_name=""):
        """
        Sampling the states at a given temperature, usually after equilibration.
        """

        # Initialize the list of sampled states.
        sample = []
        # Sample the states iteratively.
        for m_0 in range(M_sample):
            # Perform the Monte Carlo sweep many times between the sample states to make them uncorrelated.
            for m_1 in range(M_sample_diff):
                self.sweep(T, H, sweep_type=sweep_type)
            # After the sweep, add the latest state to the sample.
            sample.append(self.state.copy())
            # Print the information of current iteration, either directly or in a file.
            info = f"Sampling at T = {T} [{m_0}/{M_sample}]"
            if to_print_or_write == "print":
                print(info)
            elif to_print_or_write == "write":
                with open(file_name, "a") as file_info:
                    file_info.write(info + "\n")
        # Assign the sample to the Monte Carlo simulation.
        self.sample = sample









