/**
 * @file hamiltonian.cpp
 * @brief This class inherits the `MPO` class and constructs Hamiltonians as an `MPO`.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "hamiltonian.hpp"

HAM::HAM(Sites sites, std::string model_name, std::vector<double> params)
    : MPO(sites), model_name(model_name), params(params) {
    std::vector<Index> physical_indices = sites.get_physical_indices();

    if (model_name == "XXZ") {
        /**
         * We use MPO representation of the XXZ as decribed in page XX of U. Schollwoeck, The
         * density-matrix renormalization group in the age of matrix product states, Annals of
         * Physics 326, 96 (2011).
         *
         * H = sum^{L-1}_i [(J/2) (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+)
         *     + Delta S_i^z S_{i+1}^z] + sum_i h S_i^z
         *
         * Then, the <b'|W|b> tensors can be written as
         * I        0         0        0       0
         * S^+      0         0        0       0
         * S^-      0         0        0       0
         * S^z      0         0        0       0
         * hS^z  (J/2)S^-  (J/2)S^+  DeltaS^z  I
         *
         * On the leftmost site, we have
         * hS^z  (J/2)S^-  (J/2)S^+  DeltaS^z  I
         *
         * On the rightmost site, we have
         * I
         * S^+
         * S^-
         * S^z
         * hS^z
         *
         * To construct these W's, we look at the matrix elements given by
         * <b',s'|W|s,b> where s,s' = 0,1 and b,b' = 0,1,2,3,4
         * For example, <1|S^-|0> = 1 and <0|S^+|1> = 1.
         *
         * Our MPO's are constructed in the order of b',s',s,b and so
         * each W[i] is of the shape {b',s',s,b}:
         *
         *                  s'
         *                  |
         *              b'--W--b
         *                  |
         *                  s
         *
         * That means, W[0] (the leftmost W) is of shape {1,2,2,5},
         * W[i] (the W's in the bulk) is of shape {5,2,2,5},
         * while W[size-1] (the rightmost W) is of shape {5,2,2,1}.
         */

        post_init(5); // link_dim is 5 here

        const double J = params[0];
        const double Delta = params[1];
        const double h = params[2];

        xt::xarray<double, xt::layout_type::column_major> W_leftmost =
            xt::zeros<double>({1, 2, 2, 5});
        W_leftmost = {{
            {{h * 0.5, 0.0, 0.0, Delta * 0.5, 1.0},   // s',s = 0,0
             {0.0, 0.0, J * 0.5, 0.0, 0.0}},          // s',s = 0,1
            {{0.0, J * 0.5, 0.0, 0.0, 0.0},           // s',s = 1,0
             {-h * 0.5, 0.0, 0.0, -Delta * 0.5, 1.0}} // s',s = 1,1
        }};

        tensors[0].set_data(W_leftmost);

        xt::xarray<double, xt::layout_type::column_major> W_i = xt::zeros<double>({5, 2, 2, 5});
        W_i = {
            {
                // b' = 0 and each column is b = 0,...,5
                {{1.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 0,0
                 {0.0, 0.0, 0.0, 0.0, 0.0}}, // s',s = 0,1
                {{0.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 1,0
                 {1.0, 0.0, 0.0, 0.0, 0.0}}  // s',s = 1,1
            },

            {
                // b' = 1 and each column is b = 0,...,5
                {{0.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 0,0
                 {1.0, 0.0, 0.0, 0.0, 0.0}}, // s',s = 0,1
                {{0.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 1,0
                 {0.0, 0.0, 0.0, 0.0, 0.0}}  // s',s = 1,1
            },

            {
                // b' = 2 and each column is b = 0,...,5
                {{0.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 0,0
                 {0.0, 0.0, 0.0, 0.0, 0.0}}, // s',s = 0,1
                {{1.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 1,0
                 {0.0, 0.0, 0.0, 0.0, 0.0}}  // s',s = 1,1
            },

            {
                // b' = 3 and each column is b = 0,...,5
                {{0.5, 0.0, 0.0, 0.0, 0.0},  // s',s = 0,0
                 {0.0, 0.0, 0.0, 0.0, 0.0}}, // s',s = 0,1
                {{0.0, 0.0, 0.0, 0.0, 0.0},  // s',s = 1,0
                 {-0.5, 0.0, 0.0, 0.0, 0.0}} // s',s = 1,1
            },

            {
                // b' = 4 and each column is b = 0,...,5
                {{h * 0.5, 0.0, 0.0, Delta * 0.5, 1.0},   // s',s = 0,0
                 {0.0, 0.0, J * 0.5, 0.0, 0.0}},          // s',s = 0,1
                {{0.0, J * 0.5, 0.0, 0.0, 0.0},           // s',s = 1,0
                 {-h * 0.5, 0.0, 0.0, -Delta * 0.5, 1.0}} // s',s = 1,1
            },
        };

        for (int i = 1; i < size - 1; i++) {
            tensors[i].set_data(W_i);
        }

        xt::xarray<double, xt::layout_type::column_major> W_rightmost =
            xt::zeros<double>({1, 2, 2, 5});
        W_rightmost = {{
            // each column is b = 0,1,2,3,4
            {{1.0, 0.0, 0.0, 0.5, h * 0.5},   // s',s = 0,0
             {0.0, 1.0, 0.0, 0.0, 0.0}},      // s',s = 0,1
            {{0.0, 0.0, 1.0, 0.0, 0.0},       // s',s = 1,0
             {1.0, 0.0, 0.0, -0.5, -h * 0.5}} // s',s = 1,1
        }};

        // we need to transpose this tensor the shape {5,2,2,1}
        // by swapping the first and last axes
        W_rightmost = xt::transpose(W_rightmost, {3, 1, 2, 0});

        tensors[size - 1].set_data(W_rightmost);
    }
}

std::string HAM::get_model_name() const { return model_name; }

std::vector<double> HAM::get_params() const { return params; }
