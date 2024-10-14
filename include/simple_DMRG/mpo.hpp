/**
 * @file mpo.hpp
 * @brief Defines the `MPO` class that makes the matrix product operator using `Tensor` objects.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#ifndef MPO_HPP
#define MPO_HPP

#include "index.hpp"
#include "mps.hpp"
#include "sites.hpp"
#include "tensor.hpp"

#include <iostream>
#include <vector>

/**
 * @brief A class representing a matrix product operator (MPO).
 *
 * The `MPO` class uses `Tensor` objects to represent a many-body quantum operator as a matrix
 * product operator and provides methods for initialization and computing expectation values between
 * other matrix product states.
 */
class MPO {
public:
    /**
     * @brief Default constructor for the `MPO`.
     *
     * This initializes an empty `MPO` object.
     *
     * @param sites The `Sites` object that represents the spin sites.
     */
    MPO(Sites sites);

    /**
     * @brief Constructor for the `MPO`.
     *
     * @param sites The `Sites` object that represents the spin sites.
     * @param link_dim The initial link dimension.
     */
    MPO(Sites sites, int link_dim);

    /**
     * @brief Post facto constructor for the `MPO`.
     *
     * This re-initializes the `MPO` object using new link dimension.
     *
     * @param post_link_dim The new link dimension of the `MPO`.
     */
    void post_init(int post_link_dim = 1);

    /**
     * @brief Copy constructor for the `MPO`.
     *
     * @param other The another `MPO` object to copy from.
     */
    MPO(const MPO &other);

    /**
     * @brief Assignment operator for the `MPO`.
     *
     * @param other The other `MPO` object to assign from.
     * @return A reference to this `MPO` object.
     */
    MPO &operator=(const MPO &other);

    /**
     * @brief Gets the length of the `MPO`.
     *
     * @return The length of the `MPO` as an `int`.
     */
    int get_size() const;

    /**
     * @brief Gets the physical dimension of the `MPO`.
     *
     * @return The physical dimension as an `int`.
     */
    int get_physical_dim() const;

    /**
     * @brief Gets the `Sites` object of the `MPO`.
     *
     * @return The `Sites` object.
     */
    Sites get_sites() const;

    /**
     * @brief Gets a reference to the vector of tensors.
     *
     * @return A reference to the vector of `Tensor` objects describing the `MPO`.
     */
    std::vector<Tensor> &get_tensors();

    /**
     * @brief Gets a const reference to the vector of tensors.
     *
     * @return A const reference to the vector of `Tensor` objects describing the `MPO`.
     */
    const std::vector<Tensor> &get_tensors() const;

    /**
     * @brief Copies and gets the leftmost index.
     *
     * @return The leftmost `Index` object.
     */
    Index get_leftmost_index() const;

    /**
     * @brief Copies and gets the rightmost index.
     *
     * @return The rightmost `Index` object.
     */
    Index get_rightmost_index() const;

    /**
     * @brief Sets the elements of all the `MPO` tensors to one.
     *
     * @return A reference to this `MPO` object.
     */
    MPO &set_one();

    /**
     * @brief Sets the elements of all the `MPO` tensors to zero.
     *
     * @return A reference to this `MPO` object.
     */
    MPO &set_zero();

    /**
     * @brief Randomizes the elements of all the `MPO` tensors.
     *
     * @return A reference to this `MPO` object.
     */
    MPO &set_random();

    /**
     * @brief Gets the expectation value of this `MPO` tensor with respect to two `MPS` objects.
     *
     * This method evaluates the expectation value <lmps|MPO|rmps>.
     *
     * @param lmps The `MPS` object on the left-hand side.
     * @param rmps The `MPS` object on the right-hand side.
     * @return The expectation value as a `double`.
     */
    double get_expval(MPS &lmps, MPS &rmps) const;

protected:
    /**
     * @brief The `Sites` object for the `MPO`.
     */
    Sites sites;

    /**
     * @brief The length of the `MPO`.
     */
    int size;

    /**
     * @brief The initial link dimension of the `MPO`.
     */
    int link_dim;

    /**
     * @brief The physical dimension of the `MPO`.
     */
    int physical_dim;

    /**
     * @brief The vector of `Tensor` objects representing the `MPO`.
     */
    std::vector<Tensor> tensors;
};

#endif // MPO_HPP
