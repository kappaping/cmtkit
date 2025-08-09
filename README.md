# CMTKit: Condensed Matter Theory Toolkit
Since I started my postdoctoral stint at UC Berkeley, a significant portion of my research has been based on numerical computations of quantum many-body systems. To manage the codes for different projects harmoniously, I started to establish and maintain my own python library, CMTKit, for versatile computations in theoretical condensed matter physics. CMTKit is a scientific Python library for quantum materials and many-body lattice models. It is built upon fundamental scientific python libraries, such as NumPy, SciPy, and sparse for matrix and tensor computations, as well as Matplotlib (2D) and Mayavi (3D) for graphics. This library has supported my broad exploration into various quantum many-body systems for diverse 1D, 2D, and 3D quantum materials.

The main features of my library include:
1. Public repository on GitHub: https://github.com/kappaping/cmtkit
2. Automatic modeling of arbitrary 1D to 3D quantum lattice models [[Paper]](https://arxiv.org/abs/2406.02671)
3. Automated computation of quantum states in lattice models:
    - Variational optimization of quantum states: Hartree-Fock(-Bogoliubov) theory [[Paper]](https://arxiv.org/abs/2503.09602)
    - Time-series analysis of quantum dynamics: Time-dependent Hartree-Fock(-Bogoliubov) theory [[Paper]](https://arxiv.org/abs/2411.10447)
    - Beyond mean field: Functional renormalization group (RG) [[Reference]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.125141)
    - Monte Carlo simulation: Classical spin systems, with metropolis and Wolff cluster update
4. Tensor network methods for strongly correlated quantum systems:
    - Essential tools for tensor networks, including a direct contraction of arbitrary diagrams
    - Variational optimization of layered network ansatz: Multiscale entanglement renormalization ansatz (MERA) [[Reference]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.79.144108)
5. Automated 3D visualization of lattices and quantum states [[Paper]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.L041121)


## Demo: Altermagnetism on a honeycomb lattice

This figure of altermagnetism ([arXiv:2503.09602](https://arxiv.org/abs/2503.09602)) demonstrates the computational and graphical capabilities in my package. The upper figures show (left) the Haldane model with charge currents on the honeycomb lattice, as well as (right) its altermagnetic ground state under repulsive interaction obtained by the Hartree-Fock theory. The lower figures show (left) the band structures with spin splitting and (right) the splitting energy map of the lower two occupied bands in the Brillouin zone.

<div align="center">
  <img src="almslc.png" alt="ALM" width="600">
</div>
