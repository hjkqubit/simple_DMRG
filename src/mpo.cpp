/**
 * @file mpo.cpp
 * @brief Defines the `MPO` class that makes the matrix product operator using `Tensor` objects.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "mpo.hpp"

MPO::MPO(Sites sites) : sites(sites) {
    size = sites.get_size();
    physical_dim = sites.get_physical_dim();
}

MPO::MPO(Sites sites, int link_dim) : sites(sites), link_dim(link_dim) {
    size = sites.get_size();
    physical_dim = sites.get_physical_dim();

    std::vector<Index> physical_indices = sites.get_physical_indices();
    std::vector<Index> primed_physical_indices(size);
    for (int i = 0; i < size; i++) {
        primed_physical_indices[i] = physical_indices[i];
        primed_physical_indices[i].prime();
    }

    std::vector<Index> link_indices = sites.get_link_indices(link_dim);

    for (int i = 0; i < link_indices.size(); i++) {
        if (i == 0 || i == size) {
            link_indices[i].set_dims(1);
        }
    }

    tensors.resize(size);
    for (int i = 0; i < size; i++) {
        tensors[i] = Tensor({link_indices[i], primed_physical_indices[i], physical_indices[i],
                             link_indices[i + 1]});
    }
}

void MPO::post_init(int post_link_dim) {
    link_dim = post_link_dim;

    std::vector<Index> physical_indices = sites.get_physical_indices();
    std::vector<Index> link_indices = sites.get_link_indices(link_dim);

    std::vector<Index> primed_physical_indices(size);
    for (int i = 0; i < size; i++) {
        primed_physical_indices[i] = physical_indices[i];
        primed_physical_indices[i].prime();
    }

    for (int i = 0; i < link_indices.size(); i++) {
        if (i == 0 || i == size) {
            link_indices[i].set_dims(1);
        }
    }

    tensors.resize(size);
    for (int i = 0; i < size; i++) {
        tensors[i] = Tensor({link_indices[i], primed_physical_indices[i], physical_indices[i],
                             link_indices[i + 1]});
    }
}

MPO::MPO(const MPO &other)
    : sites(other.sites), link_dim(other.link_dim), size(other.size),
      physical_dim(other.physical_dim), tensors(other.tensors) {}

MPO &MPO::operator=(const MPO &other) {
    if (this != &other) {
        sites = other.sites;
        link_dim = other.link_dim;
        size = other.size;
        physical_dim = other.physical_dim;
        tensors = other.tensors;
    }
    return *this;
}

int MPO::get_size() const { return size; }

int MPO::get_physical_dim() const { return physical_dim; }

Sites MPO::get_sites() const { return sites; }

std::vector<Tensor> &MPO::get_tensors() { return tensors; }

const std::vector<Tensor> &MPO::get_tensors() const { return tensors; }

Index MPO::get_leftmost_index() const {
    Index left_most_index = tensors[0].get_indices()[0];
    return left_most_index;
}

Index MPO::get_rightmost_index() const {
    int end_index = tensors[size - 1].get_num_indices();
    Index right_most_index = tensors[size - 1].get_indices()[end_index - 1];
    return right_most_index;
}

MPO &MPO::set_one() {
    for (int i = 0; i < size; i++) {
        tensors[i].set_one();
    }
    return *this;
}

MPO &MPO::set_zero() {
    for (int i = 0; i < size; i++) {
        tensors[i].set_zero();
    }
    return *this;
}

MPO &MPO::set_random() {
    for (int i = 0; i < size; i++) {
        tensors[i].set_random();
    }
    return *this;
}

double MPO::get_expval(MPS &lmps, MPS &rmps) const {
    // we need to construct a Tensor object that will be used to hold the results from the tensor
    // decompositions when evaluating <lmps|MPO|rmps>.

    // start from the left and end on the right
    std::vector<Index> left_indices = {lmps.get_leftmost_index().prime(), get_leftmost_index(),
                                       rmps.get_leftmost_index()};

    Tensor result_tensor(left_indices);
    result_tensor.set_one();

    // perform tensor decomposition from left to right
    for (int i = 0; i < size; i++) {
        result_tensor *= lmps.get_tensors()[i].prime_indices();
        lmps.get_tensors()[i].unprime_indices();

        result_tensor *= tensors[i];
        result_tensor *= rmps.get_tensors()[i];
    }

    // resulting tensor is of the shape 1x1x1
    return result_tensor.get_data()(0, 0, 0);
}
