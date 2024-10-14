/**
 * @file index.cpp
 * @brief Defines the `Index` class that keeps track of physical and link dimensions.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "index.hpp"

Index::Index() : dims(1), index_tag("link"), plevel(0) { uuid_generate(id); }

Index::Index(int dims, std::string index_tag, int plevel)
    : dims(dims), index_tag(index_tag), plevel(plevel) {
    uuid_generate(id); // generate the unique identifier
}

Index::~Index() {}

Index::Index(const Index &other)
    : dims(other.dims), index_tag(other.index_tag), plevel(other.plevel) {
    uuid_copy(id, other.id); // copy the unique identifier
}

Index &Index::operator=(const Index &other) {
    if (this != &other) {
        dims = other.dims;
        index_tag = other.index_tag;
        plevel = other.plevel;
        uuid_copy(id, other.id);
    }
    return *this;
}

int Index::get_dims() const { return dims; }

std::string Index::get_index_tag() const { return index_tag; }

int Index::get_plevel() const { return plevel; }

Index &Index::set_dims(int new_dims) {
    dims = new_dims;
    return *this;
}

Index &Index::set_prime(int new_plevel) {
    plevel = new_plevel;
    return *this;
}

Index &Index::reset_prime() {
    plevel = 0;
    return *this;
}

Index &Index::prime() {
    plevel++;
    return *this;
}

Index &Index::unprime() {
    plevel--;
    return *this;
}

bool Index::operator==(const Index &rhs) const {
    return uuid_compare(id, rhs.id) == 0 && plevel == rhs.plevel;
}

std::size_t Index::hash() const {
    // combine hashes of the prime level and the UUID
    std::size_t hashValue = std::hash<int>{}(plevel);
    for (const auto &byte : id) {
        hashValue ^= std::hash<uint8_t>{}(byte);
    }
    return hashValue;
}

std::ostream &operator<<(std::ostream &os, const Index &index) {
    os << "Index(dims = " << index.dims << ", tag = " << index.index_tag
       << ", plevel = " << index.plevel << ", id = " << index.get_id_string() << ")";
    return os;
}

std::vector<std::vector<int>> find_contract_modes(const std::vector<Index> &A,
                                                  const std::vector<Index> &B) {

    // first create a dictionary with keys as the Index and values as their position `A`
    std::unordered_map<Index, int> A_map(A.size());
    std::vector<int> A_list(A.size());
    for (int i = 0; i < A.size(); i++) {
        A_map[A[i]] = i;
        A_list[i] = i; // store these positions in A_list: the contraction modes for `A`
    }

    // the last possible position for A
    int A_last = A.size() - 1;

    // get dictionary for B
    std::unordered_map<Index, int> B_map(B.size());
    // store the modes for B in `B_list`
    std::vector<int> B_list(B.size());
    // find matches in `B` by checking whether `B[j]` is a key in `A_map`
    for (int j = 0; j < B.size(); j++) {
        auto it = A_map.find(B[j]);
        if (it != A_map.end()) {
            B_list[j] = A_map[B[j]]; // match means use previously defined position
        } else {
            A_last++;
            B_list[j] = A_last; // no match means use new position starting from A_last
        }
        B_map[B[j]] = B_list[j];
    }

    // now find modes for C in `C_list`
    std::vector<int> C_list;
    std::vector<int> C_pos;
    for (int i = 0; i < A.size(); i++) {
        auto it = B_map.find(A[i]);
        if (it == B_map.end()) {
            C_list.push_back(A_list[i]); // no match in `B` means use `A`'s mode
            C_pos.push_back(i);
        }
    }
    for (int j = 0; j < B.size(); j++) {
        auto it = A_map.find(B[j]);
        if (it == A_map.end()) {
            C_list.push_back(B_list[j]); // no match in `A` means use `B`'s mode
            C_pos.push_back(j + A.size());
        }
    }

    // return `A_list`, `B_list`, `C_list` of modes and `C_pos` the position of the modes from `A`
    // and `B`. These will be used for the tensor contractions.
    return {A_list, B_list, C_list, C_pos};
}

std::string Index::get_id_string() const {
    char id_str[37];
    uuid_unparse(id, id_str);
    return std::string(id_str);
}
