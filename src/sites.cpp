/**
 * @file sites.cpp
 * @brief Defines the `Sites` class that keeps track of `Index` objects in a chain.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "sites.hpp"

Sites::Sites() {
    size = 1;         // default length is 1
    physical_dim = 2; // default physical_dim is 2

    std::string physical_tag = "physical";

    // generate the physical `Index` objects
    physical_indices.resize(size);
    for (int i = 0; i < size; i++) {
        physical_indices[i] = Index(physical_dim, physical_tag);
    }
}

Sites::Sites(int size, int physical_dim) : size(size), physical_dim(physical_dim) {
    std::string physical_tag = "physical";

    // generate the physical `Index` objects
    physical_indices.resize(size);
    for (int i = 0; i < size; i++) {
        physical_indices[i] = Index(physical_dim, physical_tag);
    }
}

int Sites::get_size() const { return size; }

int Sites::get_physical_dim() const { return physical_dim; }

std::vector<Index> Sites::get_physical_indices() const {
    // `physical_indices` are copied before the output
    std::vector<Index> copied_physical_indices(size);
    for (int i = 0; i < size; i++) {
        copied_physical_indices[i] = physical_indices[i];
    }
    return copied_physical_indices;
}

std::vector<Index> Sites::get_link_indices(int link_dim) const {
    // `link_indices` are newly generated for each call of this function
    std::vector<Index> link_indices(size + 1);
    for (int i = 0; i < size + 1; i++) {
        link_indices[i] = Index(link_dim);
    }
    return link_indices;
}
