/**
 * @file mps.hpp
 * @brief Defines the `MPS` class that makes the matrix product state using `Tensor` class objects.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#ifndef MPS_HPP
#define MPS_HPP

#include <iostream>
#include <vector>

#include "index.hpp"
#include "sites.hpp"
#include "tensor.hpp"

/**
 * @brief A class representing a matrix product state (MPS).
 *
 * The `MPS` class uses `Tensor` objects to represent a many-body quantum wavefunction as a matrix
 * product state and provides methods for state initialization, and canonicalization.
 */
class MPS {
public:
    /**
     * @brief Default constructor for `MPS`.
     */
    MPS();

    /**
     * @brief Constructor for `MPS`.
     *
     * This initializes an `MPS` object with the given `Sites` object and link dimension.
     *
     * @param sites The `Sites` object representing the spin chain.
     * @param link_dim The dimension of the links connecting each `Tensor` object.
     */
    MPS(Sites sites, int link_dim = 1);

    /**
     * @brief Copy constructor for `MPS`.
     *
     * @param other The other `MPS` object to copy from.
     */
    MPS(const MPS &other);

    /**
     * @brief Assignment operator for `MPS`.
     *
     * @param other The other `MPS` object to assign from.
     * @return A reference to this `MPS` object.
     */
    MPS &operator=(const MPS &other);

    /**
     * @brief Gets the size of the `MPS` (i.e., the length of the chain).
     *
     * @return The length of the `MPS` as an `int`.
     */
    int get_size() const;

    /**
     * @brief Gets the physical dimension of the `MPS`.
     *
     * @return The dimension of the physical Hilbert space as an `int`.
     */
    int get_physical_dim() const;

    /**
     * @brief Gets the `Sites` object for the `MPS`.
     *
     * @return The `Sites` object.
     */
    Sites get_sites() const;

    /**
     * @brief Gets a reference to the vectors of tensors describing the `MPS`.
     *
     * @return A reference to the vector of `Tensor` objects.
     */
    std::vector<Tensor> &get_tensors();

    /**
     * @brief Gets a const reference to the vectors of tensors describing the `MPS`.
     *
     * @return A const reference to the vector of `Tensor` objects.
     */
    const std::vector<Tensor> &get_tensors() const;

    /**
     * @brief Copies and gets the leftmost index of the `MPS`.
     *
     * @return A copy of the leftmost `Index` object.
     */
    Index get_leftmost_index() const;

    /**
     * @brief Copies and gets the rightmost index of the `MPS`.
     *
     * @return A copy of the rightmost `Index` object.
     */
    Index get_rightmost_index() const;

    /**
     * @brief Sets the elements of all the `MPS` tensors to one.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &set_one();

    /**
     * @brief Sets the elements of all the `MPS` tensors to zero.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &set_zero();

    /**
     * @brief Randomizes the elements of all the `MPS` tensors.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &set_random();

    /**
     * @brief Constructs the all spin-up state for the `MPS`.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &spin_up_state();

    /**
     * @brief Constructs the all spin-down state for the `MPS`.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &spin_down_state();

    /**
     * @brief Constructs the right-canonical state for the `MPS`.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &right_canonicalize();

    /**
     * @brief Constructs the left-canonical state for the `MPS`.
     *
     * @return A reference to this `MPS` object.
     */
    MPS &left_canonicalize();

private:
    /**
     * @brief The `Sites` object for the `MPS`.
     */
    Sites sites;

    /**
     * @brief The length of the `MPS`.
     */
    int size;

    /**
     * @brief The initial link dimension of the `MPS`.
     */
    int link_dim;

    /**
     * @brief The physical dimension of the `MPS`.
     */
    int physical_dim;

    /**
     * @brief The vector of `Tensor` objects representing the `MPS`.
     */
    std::vector<Tensor> tensors;
};

#endif // MPS_HPP
