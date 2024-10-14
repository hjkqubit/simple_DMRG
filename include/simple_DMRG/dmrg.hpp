/**
 * @file dmrg.hpp
 * @brief This class implements the density matrix renormalization group algorithm.
 *
 * This header defines the `DMRG` class, which includes methods to perform the DMRG algorithm.
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#ifndef DMRG_HPP
#define DMRG_HPP

#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <xtensor/xbuilder.hpp>

#include "index.hpp"
#include "mpo.hpp"
#include "mps.hpp"
#include "sites.hpp"
#include "tensor.hpp"

/**
 * @brief A structure that sets the parameters of the DMRG algorithm.
 */
struct solver_params {
    int maxdim = 100;          // maximum link dimension for truncation
    double cutoff = 1e-12;     // truncation cutoff value for discarding singular values
    int lanczos_krylovdim = 3; // dimension of the Krylov subspace during the Lanczos method
    int lanczos_maxiter = 1;   // the number of repeats of the Lanczos algorithm
};

/**
 * @brief A class that implements the DMRG algorithm.
 *
 * The `DMRG` class performs the density matrix renormalization group (DMRG) method to compute the
 * ground state and energy of a many-body quantum spin system. This class uses both matrix product
 * states and operators to perform DMRG.
 */
class DMRG {
public:
    /**
     * @brief Constructs a `DMRG` object with an initial `MPS` state.
     *
     * This constructor initializes the DMRG algorithm by constructing the L and R blocks.
     *
     * @param sites The lattice of spins, represented by the `Sites` object.
     * @param H The Hamiltonian as an `MPO` object.
     * @param psi The initial ansatz state as an `MPS` object.
     */
    DMRG(Sites sites, MPO &H, MPS psi);

    /**
     * @brief Constructs a `DMRG` object.
     *
     * This constructor uses a randomly initialized `MPS` as its initial ansatz for the DMRG method.
     * Further, this constructor initializes the DMRG algorithm by constructing the L and R blocks.
     *
     * @param sites The lattice of spins, represented by the `Sites` object.
     * @param H The Hamiltonian as an `MPO` object.
     */
    DMRG(Sites sites, MPO &H);

    /**
     * @brief Performs the DMRG algorithm.
     *
     * This function runs the DMRG algorithm for a specified number of sweeps and solver parameters.
     *
     * @param nsweeps The total number of DMRG sweeps to take.
     * @param dmrg_params The paramters for controlling the DMRG procedure.
     * @return A tuple containing the ground state energy and the optimized MPS.
     */
    std::tuple<double, MPS> dmrg(int nsweeps, const solver_params &dmrg_params);

private:
    /**
     * @brief The lattice of spins
     */
    Sites sites;

    /**
     * @brief The Hamiltonian as an `MPO`.
     */
    MPO H;

    /**
     * @brief The initial/optimized state as an `MPS`.
     */
    MPS psi;

    /**
     * @brief The length of the spin chain.
     */
    int size;

    /**
     * @brief The left blocks.
     */
    std::unordered_map<int, Tensor> L_blocks;

    /**
     * @brief The right blocks.
     */
    std::unordered_map<int, Tensor> R_blocks;

    /**
     * @brief Builds the left and right blocks.
     *
     * This method constructs the left (`L_blocks`) and right (`R_blocks`) blocks during the
     * initialization of the `DMRG` object.
     */
    void build_LR_blocks();

    /**
     * @brief Performs the Lanczos algorithm.
     *
     * Applies the Lanczos algorithm to a given tensor to obtain the approximate lowest eigenvalue.
     * The operator of concern is l_block * M * r_block, where M is the tensor to be optimized.
     *
     * @param M The current tensor to be optimized.
     * @param l_block The left block.
     * @param r_block The right block.
     * @param lanczos_params The parameters for the Lanczos algorithm: krylov_dim and maxiter.
     * @return The lowest eigenvalue.
     */
    double perform_lanczos(Tensor &M, Tensor &l_block, Tensor &r_block,
                           const solver_params &lanczos_params) const;

    /**
     * @brief Performs a single DMRG sweep.
     *
     * Optimizes the wavefunction by sweeping from left to right or right to left.
     *
     * @param i The current step of the DMRG sweep.
     * @param sweep_right Whether the sweep is from left to right or right to left.
     * @param dmrg_params The parameters for controlling the DMRG procedure.
     * @return A tuple containing the updated energy and the maximum link dimension of the MPS.
     */
    std::tuple<double, int> dmrg_sweep(int i, bool sweep_right, const solver_params &dmrg_params);

    /**
     * @brief Calculates the L2 norm of the tensor.
     *
     * @param tensor The tensor for which to compute the norm.
     * @return The L2 norm of the tensor.
     */
    double get_xt_norm(xt::xarray<double, xt::layout_type::column_major> &tensor) const;
};

#endif // DMRG_HPP
