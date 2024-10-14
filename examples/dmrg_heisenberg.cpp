/**
 * @file dmrg_heisenberg.cpp
 * @brief A sample code to run the DMRG algorithm on the Heisenberg model
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "dmrg.hpp"
#include "hamiltonian.hpp"
#include "mps.hpp"
#include "sites.hpp"

int main(int argc, char *argv[]) {

    // define the default lattice sites
    int L = 10;           // length of the spin chain
    int physical_dim = 2; // spin-1/2 particles

    // define the default model
    std::string model_name = "XXZ"; // use XXZ model

    // define the default coupling parameters {J, Delta, h}
    double J = 1.0;
    double Delta = 1.0;
    double h = 0.0;

    // define the default DMRG parameter
    int nsweeps = 10;      // number of DMRG sweeps
    int maxdim = 100;      // maximum link dimension
    double cutoff = 1e-10; // truncation cutoff error

    // get any user-defined parameters
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            std::string arg_val(argv[i]);

            if (arg_val == "-L" and i < argc - 1) {
                L = std::stoi(argv[i + 1]);
                i++;
            } else if (arg_val == "-J" and i < argc - 1) {
                J = std::stod(argv[i + 1]);
                i++;
            } else if (arg_val == "-Delta" and i < argc - 1) {
                Delta = std::stod(argv[i + 1]);
                i++;
            } else if (arg_val == "-h" and i < argc - 1) {
                h = std::stod(argv[i + 1]);
                i++;
            } else if (arg_val == "-nsweeps" and i < argc - 1) {
                nsweeps = std::stoi(argv[i + 1]);
                i++;
            } else if (arg_val == "-maxdim" and i < argc - 1) {
                maxdim = std::stoi(argv[i + 1]);
                i++;
            } else if (arg_val == "-cutoff" and i < argc - 1) {
                cutoff = std::stod(argv[i + 1]);
                i++;
            }
        }
    }

    // set up the lattice sites
    Sites sites(L, physical_dim);

    // set up the Hamiltonian
    std::vector<double> params = {J, Delta, h};
    HAM H(sites, model_name, params);

    // set up DMRG
    DMRG dmrg(sites, H);
    solver_params default_params;
    default_params.maxdim = maxdim;
    default_params.cutoff = cutoff;

    // run DMRG
    std::tuple<double, MPS> res = dmrg.dmrg(nsweeps, default_params);

    // get DMRG results
    double energy = std::get<0>(res); // ground state energy
    MPS psi = std::get<1>(res);       // ground state MPS

    return 0;
}
