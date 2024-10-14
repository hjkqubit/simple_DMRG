# simple_DMRG
A simple implementation of the density matrix renormalization group (DMRG) in C++ using CUDA.

This repository follows the descriptions of DMRG in [U. Schollwoeck, The density-matrix renormalization group in the age of matrix product states, Annals of Physics 326, 96 (2011)] and in [Tensor Networks Website](https://tensornetwork.org/mps/algorithms/dmrg/). These implementations of the DMRG algorithm use the matrix product states and operators to represent many-body quantum wavefunctions and operators, respectively. The structure of this code, such as using `Index` and `Sites` objects to describe spin lattices, is inspired by [ITensors.jl](https://github.com/ITensor/ITensors.jl). 

This project takes advantage of CUDA parallel computing by performing tensor contractions (using cuTENSOR) and singular value decompositions (using cuSOLVER) on the GPU.

Please note that this is a simple implementation of DMRG and thus has not been rigorously tested nor optimized!

## Installation

To build and run this project, one must have the following dependencies installed:
- CMAKE (version 3.19 or higher)
- CUDA Toolkit
- BLAS and LAPACK libraries
- C++11 compatible compiler
- [cuTENSOR](https://developer.nvidia.com/cutensor)
- [xtl](https://github.com/xtensor-stack/xtl) and [xtensor](https://github.com/xtensor-stack/xtensor)
- [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas)
- `libuuid` (for UUID support)

In the `CMAKELists.txt` file, make sure to properly set `CUTENSOR_ROOT` and the paths for xtl, xtensor, xtensor-blas libraries.

Clone the repository and create a build directory.

```bash
git clone https://github.com/hjkqubit/simple_DMRG.git
cd simple_DMRG
mkdir build
cd build
cmake ..
make
```

This will compile the example code `dmrg_heisenerg.cpp` in the `examples` directory.

## Usage

After building the project, one can run the example code `dmrg_heisenerg.cpp` from the `examples` directory. To do this, first access the `build/examples` directory and run `./dmrg_heisenberg`.

```bash
./dmrg_heisenberg
```

This performs DMRG on the Hamiltonian defined as

$$H = \sum^{L-1}_{i=1}\left(\frac{J}{2}\left(S^+_i S\_{i+1}^- + S^-_i S\_{i+1}^+ \right) + \Delta S_i^z S\_{i+1}^z\right) + h \sum^{L}\_{i=1} S_i^z.$$

The default parameters for `dmrg_heisenberg.cpp` are `L = 10`, `J = Delta = 1`, `h = 0`. Running `./dmrg_heisenberg` may output something similar to the following:
```bash
sweep: 0, energy = -3.94379, dim = 4
sweep: 1, energy = -4.258, dim = 16
sweep: 2, energy = -4.25804, dim = 21
sweep: 3, energy = -4.25804, dim = 21
sweep: 4, energy = -4.25804, dim = 21
sweep: 5, energy = -4.25804, dim = 21
sweep: 6, energy = -4.25804, dim = 21
sweep: 7, energy = -4.25804, dim = 21
sweep: 8, energy = -4.25804, dim = 21
sweep: 9, energy = -4.25804, dim = 21
```

One can change the parameters of the model and DMRG procedure as the following:
```bash
./dmrg_heisenberg -L <L_value> -J <J_value> -Delta <Delta_value> -h <h_value> -nsweeps <nsweeps> -maxdim <maxdim> -cutoff <cutoff>
```

For example, we can look at the XXZ model with anisotropy (`J = 1`, `Delta = 1.5`, and `h = 0`) with `L = 100`:
```bash
./dmrg_heisenberg -L 100 -J 1 -Delta 1.5 -h 0 -nsweeps 10 -maxdim 100 -cutoff 1e-9
sweep: 0, energy = -51.3379, dim = 4
sweep: 1, energy = -52.0699, dim = 16
sweep: 2, energy = -52.0759, dim = 42
sweep: 3, energy = -52.0775, dim = 60
sweep: 4, energy = -52.079, dim = 72
sweep: 5, energy = -52.0795, dim = 77
sweep: 6, energy = -52.0796, dim = 87
sweep: 7, energy = -52.0796, dim = 100
sweep: 8, energy = -52.0796, dim = 100
sweep: 9, energy = -52.0796, dim = 100
```
