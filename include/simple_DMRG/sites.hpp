/**
 * @file sites.hpp
 * @brief Defines the `Sites` class that keeps track of `Index` objects in a chain.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#ifndef SITES_HPP
#define SITES_HPP

#include "index.hpp"

/**
 * @brief A class that represents a collection of `Index` objects in a chain.
 *
 * This class manages a collection of `Index` objects, which represents a spin chain. The `Sites`
 * class holds two types of `Index` objects: physical and link.
 */
class Sites {
public:
    /**
     * @brief Default constructor for `Sites` class.
     *
     * This constructor initializes an empty `Sites` object with default values.
     */
    Sites();

    /**
     * @brief Constructor for `Sites`.
     *
     * This constructor initializes a `Sites` object with specific length and physical dimension.
     *
     * @param size The length of the spin chain.
     * @param physical_dim The dimension of the physical indices (default is 2 for spin-1/2).
     */
    Sites(int size, int physical_dim = 2);

    /**
     * @brief Gets the size of `Sites`.
     *
     * @return The number of `Index` objects in this `Sites` object as an `int`.
     */
    int get_size() const;

    /**
     * @brief Gets the dimension of the physical Hilbert space.
     *
     * @return The dimension of the physical `Index` object as an `int`.
     */
    int get_physical_dim() const;

    /**
     * @brief Gets the vector of the physical `Index` objects.
     *
     * @return The vector of physical `Index` objects in this `Sites` object.
     */
    std::vector<Index> get_physical_indices() const;

    /**
     * @brief Gets the vector of the link `Index` objects.
     *
     * @return The vector of link `Index` objects in this `Sites` object.
     */
    std::vector<Index> get_link_indices(int link_dim = 1) const;

private:
    /**
     * @brief The number of `Index` objects in the collection: the length of the spin chain.
     */
    int size;

    /**
     * @brief The physical dimension of the indices.
     */
    int physical_dim;

    /**
     * @brief The vectors of the physical `Indext` objects.
     */
    std::vector<Index> physical_indices;
};

#endif // SITES_HPP
