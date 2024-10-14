/**
 * @file dmrg.cpp
 * @brief This class implements the density matrix renormalization group algorithm.
 *
 * This header defines the `DMRG` class, which includes methods to perform the DMRG algorithm.
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#include "dmrg.hpp"

DMRG::DMRG(Sites sites, MPO &H, MPS psi) : sites(sites), H(H), psi(psi.right_canonicalize()) {
    size = sites.get_size();

    build_LR_blocks(); // initialize the left and right blocks
}

DMRG::DMRG(Sites sites, MPO &H) : sites(sites), H(H) {
    size = sites.get_size();

    MPS phi(sites, 1);

    phi.set_random().right_canonicalize(); // right canonicalize the state
    psi = phi;

    build_LR_blocks(); // initialize the left and right blocks
}

void DMRG::build_LR_blocks() {

    L_blocks[0] =
        Tensor({psi.get_leftmost_index().prime(), H.get_leftmost_index(), psi.get_leftmost_index()})
            .set_one();

    R_blocks[size - 1] = Tensor({psi.get_rightmost_index().prime(), H.get_rightmost_index(),
                                 psi.get_rightmost_index()})
                             .set_one();

    for (int i = size - 2; i >= 0; i--) {
        R_blocks[i] = psi.get_tensors()[i + 1].prime_indices() * R_blocks[i + 1];
        R_blocks[i] *= H.get_tensors()[i + 1];
        R_blocks[i] *= psi.get_tensors()[i + 1].unprime_indices();
    }
}

double DMRG::perform_lanczos(Tensor &M, Tensor &l_block, Tensor &r_block,
                             const solver_params &lanczos_params) const {
    // get solver_params
    int krylovdim = lanczos_params.lanczos_krylovdim;
    int maxiter = lanczos_params.lanczos_maxiter;

    // shape of M tensor
    auto M_shape = M.get_data().shape();

    // size of M tensor
    int M_size = M.get_data().size();

    // dummy tensors X and Y to be used
    Tensor X = M;

    // Orthonormal Krylov vectors |v>
    xt::xarray<double, xt::layout_type::column_major> v_prev = xt::zeros<double>(M_shape);
    xt::xarray<double, xt::layout_type::column_major> v_next = M.get_data();

    // Storing a Krylov vectors
    xt::xarray<double, xt::layout_type::column_major> V_matrix =
        xt::zeros<double>({M_size, krylovdim});

    // storing alpha and beta's
    xt::xarray<double, xt::layout_type::column_major> alphas = xt::zeros<double>({krylovdim});
    xt::xarray<double, xt::layout_type::column_major> betas = xt::zeros<double>({krylovdim + 1});

    // w vectors
    xt::xarray<double, xt::layout_type::column_major> w = xt::zeros<double>(M_shape);

    // storing the triagular matrix
    xt::xarray<double, xt::layout_type::column_major> H_triag =
        xt::zeros<double>({krylovdim, krylovdim});

    // storing the eigenvalues and eigenvectors
    xt::xarray<double, xt::layout_type::column_major> eigenvalues = xt::zeros<double>({krylovdim});
    xt::xarray<double, xt::layout_type::column_major> eigenvectors =
        xt::zeros<double>({krylovdim, krylovdim});

    // save energy
    double energy;

    // perform the Lanczos algorithm
    for (int it = 0; it < maxiter; it++) {

        betas(0) = 0.0;
        v_prev = xt::zeros<double>(M_shape);
        V_matrix = xt::zeros<double>({M_size, krylovdim});
        v_next /= get_xt_norm(v_next);

        for (int i = 0; i < krylovdim; i++) {

            // w_i = A|v_i>
            X.set_data(v_next);
            X = l_block * X;
            X *= r_block;
            w = X.get_data();

            // alpha_i = <v_i|w_i>
            alphas(i) = xt::linalg::dot(xt::reshape_view(w, {M_size}),
                                        xt::reshape_view(v_next, {M_size}))(0);

            // |w_i> = |w_i> - beta_{i-1} |v_{i-1}> - alpha_i |v_i>
            w -= (betas(i) * v_prev + alphas(i) * v_next);

            // beta_i = sqrt{<w_i|w_i>}
            betas(i + 1) = get_xt_norm(w);

            // copy v_prev = v_next
            v_prev = v_next;

            // |v_i> = w / beta_i;
            v_next = w / betas(i + 1);

            // save v_next in V_matrix
            xt::view(V_matrix, xt::all(), i) = xt::reshape_view(v_prev, {M_size});
        }

        // construct the diagonal matrix denoted H_triag
        H_triag = xt::diag(alphas);

        for (int i = 0; i < krylovdim - 1; i++) {
            H_triag(i + 1, i) = betas(i + 1); // lower triangular matrix
        }

        // diagonalize H_triag using eigh
        auto result_eigh = xt::linalg::eigh(H_triag);

        eigenvalues = std::get<0>(result_eigh);
        eigenvectors = std::get<1>(result_eigh);

        energy = eigenvalues(0);
        v_next = xt::reshape_view(xt::linalg::dot(V_matrix, xt::col(eigenvectors, 0)), M_shape);
    }

    M.set_data(v_next);

    return energy;
}

double DMRG::get_xt_norm(xt::xarray<double, xt::layout_type::column_major> &tensor) const {
    auto norm_sq = xt::norm_sq(tensor);
    double norm = std::sqrt(xt::sum(norm_sq)());

    return norm;
}

std::tuple<double, int> DMRG::dmrg_sweep(int i, bool sweep_right,
                                         const solver_params &dmrg_params) {
    double cutoff = dmrg_params.cutoff;
    int maxdim = dmrg_params.maxdim;

    Tensor M = psi.get_tensors()[i] * psi.get_tensors()[i + 1]; // look at i,i+1 sites

    Tensor l_block = L_blocks[i] * H.get_tensors()[i];
    Tensor r_block = H.get_tensors()[i + 1] * R_blocks[i + 1];

    // `perform_lanczos` modifies `M`
    double energy = perform_lanczos(M, l_block, r_block, dmrg_params);

    // do truncation via SVD on tensor `M`
    std::vector<int> l = {0, 1};
    std::vector<int> r = {2, 3};

    std::tuple<Tensor, Tensor, xt::xarray<double, xt::layout_type::column_major>, int, double>
        svd_result = M.svd(l, r, sweep_right, cutoff, 1, maxdim); // mindim = 1

    // use the truncated tensors
    psi.get_tensors()[i] = std::get<0>(svd_result);
    psi.get_tensors()[i + 1] = std::get<1>(svd_result);
    int dim = std::get<3>(svd_result);

    // update the `L_blocks` or `R_blocks`
    if (sweep_right) {
        L_blocks[i + 1] = L_blocks[i] * psi.get_tensors()[i].prime_indices();
        L_blocks[i + 1] *= H.get_tensors()[i];
        L_blocks[i + 1] *= psi.get_tensors()[i].unprime_indices();
    } else {
        R_blocks[i] = psi.get_tensors()[i + 1].prime_indices() * R_blocks[i + 1];
        R_blocks[i] *= H.get_tensors()[i + 1];
        R_blocks[i] *= psi.get_tensors()[i + 1].unprime_indices();
    }

    return std::tuple<double, int>(energy, dim);
}

std::tuple<double, MPS> DMRG::dmrg(int nsweeps, const solver_params &dmrg_params) {
    std::tuple<double, int> dmrg_result;
    double energy; // the ground state energy
    int dim = 0;   // maximum bond dimension of the optimized MPS
    int dim_sweep; // dummy variable to get bond dimension from each DMRG sweep

    for (int i = 0; i < nsweeps; i++) {
        for (int j = 0; j < size - 1; j++) {
            dmrg_result = dmrg_sweep(j, true, dmrg_params);
            energy = std::get<0>(dmrg_result);
            dim_sweep = std::get<1>(dmrg_result);
            dim = std::max(dim, dim_sweep);
        }
        for (int j = size - 2; j >= 0; j--) {
            dmrg_result = dmrg_sweep(j, false, dmrg_params);
            energy = std::get<0>(dmrg_result);
            dim_sweep = std::get<1>(dmrg_result);
            dim = std::max(dim, dim_sweep);
        }

        std::cout << "sweep: " << i << ", energy = " << energy << ", dim = " << dim << std::endl;
    }

    return std::tuple<double, MPS>(energy, psi);
}
