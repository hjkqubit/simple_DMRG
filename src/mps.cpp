/**
 * @file mps.cpp
 * @brief Defines the `MPS` class that makes the matrix product state using `Tensor` class objects.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "mps.hpp"

MPS::MPS() {
    sites = Sites();
    size = sites.get_size();
    link_dim = 1;
    physical_dim = sites.get_physical_dim();

    std::vector<Index> physical_indices = sites.get_physical_indices();
    std::vector<Index> link_indices = sites.get_link_indices(link_dim);

    for (int i = 0; i < link_indices.size(); i++) {
        if (i == 0 || i == size) {
            link_indices[i].set_dims(1);
        }
    }

    tensors.resize(size);
    for (int i = 0; i < size; i++) {
        tensors[i] = Tensor({link_indices[i], physical_indices[i], link_indices[i + 1]});
    }
}

MPS::MPS(Sites sites, int link_dim) : sites(sites), link_dim(link_dim) {
    size = sites.get_size();
    physical_dim = sites.get_physical_dim();

    std::vector<Index> physical_indices = sites.get_physical_indices();
    std::vector<Index> link_indices = sites.get_link_indices(link_dim);

    for (int i = 0; i < link_indices.size(); i++) {
        if (i == 0 || i == size) {
            link_indices[i].set_dims(1); // dummy link indices with dimension 1
        }
    }

    tensors.resize(size);
    for (int i = 0; i < size; i++) {
        tensors[i] = Tensor({link_indices[i], physical_indices[i], link_indices[i + 1]});
    }
}

MPS::MPS(const MPS &other)
    : sites(other.sites), link_dim(other.link_dim), size(other.size),
      physical_dim(other.physical_dim), tensors(other.tensors) {}

MPS &MPS::operator=(const MPS &other) {
    if (this != &other) {
        sites = other.sites;
        link_dim = other.link_dim;
        size = other.size;
        physical_dim = other.physical_dim;
        tensors = other.tensors;
    }
    return *this;
}

int MPS::get_size() const { return size; }

int MPS::get_physical_dim() const { return physical_dim; }

Sites MPS::get_sites() const { return sites; }

std::vector<Tensor> &MPS::get_tensors() { return tensors; }

const std::vector<Tensor> &MPS::get_tensors() const { return tensors; }

Index MPS::get_leftmost_index() const {
    Index left_most_index = tensors[0].get_indices()[0]; // copy the leftmost index
    return left_most_index;
}

Index MPS::get_rightmost_index() const {
    int end_index = tensors[size - 1].get_num_indices();
    Index right_most_index =
        tensors[size - 1].get_indices()[end_index - 1]; // copy the rightmost index
    return right_most_index;
}

MPS &MPS::set_one() {
    for (int i = 0; i < size; i++) {
        tensors[i].set_one();
    }
    return *this;
}

MPS &MPS::set_zero() {
    for (int i = 0; i < size; i++) {
        tensors[i].set_zero();
    }
    return *this;
}

MPS &MPS::set_random() {
    for (int i = 0; i < size; i++) {
        tensors[i].set_random();
    }
    return *this;
}

MPS &MPS::spin_up_state() {
    if (physical_dim != 2) {
        std::cerr << "Error in " << __FILE__ << ": physical_dim needs to be 2" << std::endl;
        std::exit(EXIT_FAILURE);
    } else if (link_dim != 1) {
        std::cerr << "Error in " << __FILE__ << ": link_dim needs to be 1" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    xt::xarray<double, xt::layout_type::column_major> up_tensor = xt::zeros<double>({1, 2, 1});
    up_tensor(0, 0, 0) = 1.0; // spin-up state

    for (int i = 0; i < size; i++) {
        tensors[i].set_data(up_tensor);
    }
    return *this;
}

MPS &MPS::spin_down_state() {
    if (physical_dim != 2) {
        std::cerr << "Error in " << __FILE__ << ": physical_dim needs to be 2" << std::endl;
        std::exit(EXIT_FAILURE);
    } else if (link_dim != 1) {
        std::cerr << "Error in " << __FILE__ << ": link_dim needs to be 1" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    xt::xarray<double, xt::layout_type::column_major> up_tensor = xt::zeros<double>({1, 2, 1});
    up_tensor(0, 1, 0) = 1.0; // spin-down state

    for (int i = 0; i < size; i++) {
        tensors[i].set_data(up_tensor);
    }
    return *this;
}

MPS &MPS::right_canonicalize() {
    std::vector<int> l = {0};
    std::vector<int> r = {1, 2};

    for (int i = size - 1; i > 0; i--) {
        auto svd_result = (tensors[i]).svd(l, r, false);
        Tensor M_tilde = std::get<0>(svd_result);
        Tensor B = std::get<1>(svd_result);

        tensors[i] = B;
        tensors[i - 1] *= M_tilde;
    }

    tensors[0].normalize();
    return *this;
}

MPS &MPS::left_canonicalize() {
    std::vector<int> l = {0, 1};
    std::vector<int> r = {2};

    for (int i = 0; i < size - 2; i++) {
        auto svd_result = (tensors[i]).svd(l, r, true);
        Tensor A = std::get<0>(svd_result);
        Tensor M_tilde = std::get<1>(svd_result);

        tensors[i] = A;
        tensors[i - 1] = M_tilde * tensors[i - 1];
    }

    tensors[size - 1].normalize();
    return *this;
}
