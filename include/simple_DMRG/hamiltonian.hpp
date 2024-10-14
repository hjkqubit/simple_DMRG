/**
 * @file hamiltonian.hpp
 * @brief This class inherits the `MPO` class and constructs Hamiltonians as an `MPO`.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#ifndef HAMILTONIAN_HPP
#define HAMILTONIAN_HPP

#include <iostream>
#include <vector>

#include "index.hpp"
#include "mpo.hpp"
#include "sites.hpp"
#include "tensor.hpp"

/**
 * @brief A class that constructs user-specified many-body quantum Hamiltonians as `MPO` objects.
 *
 * The `HAM` class inherits the `MPO` class.
 */
class HAM : public MPO {
public:
    /**
     * @brief Constructor for the `HAM` class.
     *
     * @param sites The `Sites` object.
     * @param model_name The `std::string` that denotes the model.
     * @param params The coupling parameters of the model.
     */
    HAM(Sites sites, std::string model_name, std::vector<double> params);

    /**
     * @brief Gets the model's name.
     *
     * @return The model's name as a `std::string`.
     */
    std::string get_model_name() const;

    /**
     * @brief Gets the paramters of the model.
     *
     * @return The model's parameters as a `std::vector<double>`.
     */
    std::vector<double> get_params() const;

private:
    /**
     * @brief The model's name.
     */
    std::string model_name;

    /**
     * @brief The model's parameters.
     */
    std::vector<double> params;
};

#endif // HAMILTONIAN_HPP
