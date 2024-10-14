/**
 * @file tensor.cpp
 * @brief Defines the `Tensor` class that uses `Index` to represent a single tensor object
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 0.0
 */

#include "tensor.hpp"

// handle cuTENSOR errors
#define HANDLE_ERROR(x)                                                                            \
    {                                                                                              \
        const auto err = x;                                                                        \
        if (err != CUTENSOR_STATUS_SUCCESS) {                                                      \
            printf("Error: %s\n", cutensorGetErrorString(err));                                    \
            exit(-1);                                                                              \
        }                                                                                          \
    };

// CUDA error checks
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << cudaGetErrorString(err) << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

Tensor::Tensor() {
    indices = {Index()};
    num_indices = indices.size();
    set_zero();
}

Tensor::Tensor(std::vector<Index> indices) : indices(indices), num_indices(indices.size()) {
    set_zero(); // set all elements of tensor data to zero
}

Tensor::Tensor(std::vector<Index> indices,
               const xt::xarray<double, xt::layout_type::column_major> &set_data)
    : indices(indices), num_indices(indices.size()) {
    if (num_indices != set_data.shape().size()) {
        std::cerr << "Error in " << __FILE__
                  << ": the number of indices do not match. `num_indices` = " << num_indices
                  << ", but got " << set_data.shape().size() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    data = set_data;
}

Tensor::~Tensor() {}

Tensor::Tensor(const Tensor &other)
    : indices(other.indices), num_indices(other.num_indices), data(other.data) {}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        indices = other.indices;
        num_indices = other.num_indices;
        data = other.data;
    }
    return *this;
}

Tensor &Tensor::set_one() {
    std::vector<size_t> shape(num_indices);
    for (int i = 0; i < num_indices; i++) {
        shape[i] = indices[i].get_dims();
    }
    data = xt::ones<double>(shape);
    return *this;
}

Tensor &Tensor::set_zero() {
    std::vector<size_t> shape(num_indices);
    for (int i = 0; i < num_indices; i++) {
        shape[i] = indices[i].get_dims();
    }
    data = xt::zeros<double>(shape);
    return *this;
}

Tensor &Tensor::set_random() {
    std::vector<size_t> shape(num_indices);
    for (int i = 0; i < num_indices; i++) {
        shape[i] = indices[i].get_dims();
    }
    data = xt::random::rand<double>(shape);
    return *this;
}

xt::xarray<double, xt::layout_type::column_major> Tensor::get_data() const { return data; }

std::vector<Index> Tensor::get_indices() const { return indices; }

int Tensor::get_num_indices() const { return num_indices; }

Tensor &Tensor::set_data(const xt::xarray<double, xt::layout_type::column_major> &new_data) {
    if (data.shape() != new_data.shape()) {
        std::cerr << "Error in " << __FILE__ << ": the shapes of `data` and `new_data` do not match"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        data = new_data;
    }

    return *this;
}

Tensor &Tensor::prime_indices() {
    for (int i = 0; i < num_indices; i++) {
        indices[i].prime();
    }
    return *this;
}

Tensor &Tensor::unprime_indices() {
    for (int i = 0; i < num_indices; i++) {
        indices[i].unprime();
    }
    return *this;
}

Tensor &Tensor::normalize() {
    auto norm_sq = xt::norm_sq(data);
    double norm = xt::sum(norm_sq)();
    norm = std::sqrt(norm);

    data /= norm;
    return *this;
}

double Tensor::get_norm() const {
    auto norm_sq = xt::norm_sq(data);
    double norm = xt::sum(norm_sq)();
    norm = std::sqrt(norm);

    return norm;
}

Tensor Tensor::operator+(const double &scalar) const {
    Tensor add_tensor = Tensor(*this);
    add_tensor.data += scalar;
    return add_tensor;
}

Tensor &Tensor::operator+=(const double &scalar) {
    data += scalar;
    return *this;
}

Tensor Tensor::operator-(const double &scalar) const {
    Tensor sub_tensor = Tensor(*this);
    sub_tensor.data -= scalar;
    return sub_tensor;
}

Tensor &Tensor::operator-=(const double &scalar) {
    data -= scalar;
    return *this;
}

Tensor Tensor::operator*(const double &scalar) const {
    Tensor mul_tensor = Tensor(*this);
    mul_tensor.data *= scalar;
    return mul_tensor;
}

Tensor &Tensor::operator*=(const double &scalar) {
    data *= scalar;
    return *this;
}

Tensor Tensor::operator+(const Tensor &other) const {
    Tensor add_tensor = Tensor(*this);
    add_tensor.data += other.data;
    return add_tensor;
}

Tensor &Tensor::operator+=(const Tensor &other) {
    data += other.data;
    return *this;
}

Tensor Tensor::operator-(const Tensor &other) const {
    Tensor sub_tensor = Tensor(*this);
    sub_tensor.data -= other.data;
    return sub_tensor;
}

Tensor &Tensor::operator-=(const Tensor &other) {
    data -= other.data;
    return *this;
}

Tensor Tensor::operator*(const Tensor &other) const {
    std::vector<std::vector<int>> contract_modes = find_contract_modes(indices, other.indices);

    // A refers to `this`, B refers to `other`, and C refers to the resulting tensor from the
    // multiplication

    // number of modes
    std::vector<int> modeA = contract_modes[0];
    std::vector<int> modeB = contract_modes[1];
    std::vector<int> modeC = contract_modes[2];
    std::vector<int> posC = contract_modes[3];
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // create the vectors of extents for each tensor and the number of elements
    std::vector<int64_t> extentA(nmodeA);
    size_t elementsA = 1;
    for (int i = 0; i < nmodeA; i++) {
        extentA[i] = indices[i].get_dims();
        elementsA *= extentA[i];
    }
    size_t elementsB = 1;
    std::vector<int64_t> extentB(nmodeB);
    for (int i = 0; i < nmodeB; i++) {
        extentB[i] = other.indices[i].get_dims();
        elementsB *= extentB[i];
    }
    size_t elementsC = 1;
    std::vector<int64_t> extentC(nmodeC);
    std::vector<size_t> shapeC(nmodeC);
    std::vector<Index> indicesC(nmodeC);
    for (int i = 0; i < nmodeC; i++) {
        int pos = posC[i];
        if (pos < nmodeA) {
            extentC[i] = indices[pos].get_dims();
            indicesC[i] = indices[pos];
        } else {
            extentC[i] = other.indices[pos - nmodeA].get_dims();
            indicesC[i] = other.indices[pos - nmodeA];
        }
        shapeC[i] = extentC[i];
        elementsC *= extentC[i];
    }

    // get size in bytes
    size_t sizeA = sizeof(double) * elementsA;
    size_t sizeB = sizeof(double) * elementsB;
    size_t sizeC = sizeof(double) * elementsC;

    // allocate on device
    void *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, sizeA);
    cudaMalloc((void **)&B_d, sizeB);
    cudaMalloc((void **)&C_d, sizeC);

    // make empty xarray for C
    xt::xarray<double, xt::layout_type::column_major> C_data = xt::zeros<double>(shapeC);

    // copy to device
    CUDA_CHECK(cudaMemcpy(A_d, data.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, (other.data).data(), sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_data.data(), sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    // perform cuTENSOR's tensor contraction
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    // create the tensor descriptions for A, B, and C
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, nmodeA, extentA.data(), NULL,
                                                CUTENSOR_R_64F, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, nmodeB, extentB.data(), NULL,
                                                CUTENSOR_R_64F, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, nmodeC, extentC.data(), NULL,
                                                CUTENSOR_R_64F, kAlignment));

    // create the tensor description for the contraction
    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                                           descB, modeB.data(), CUTENSOR_OP_IDENTITY, descC,
                                           modeC.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(),
                                           CUTENSOR_COMPUTE_DESC_64F));

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void *)&scalarType, sizeof(scalarType)));

    // use double
    assert(scalarType == CUTENSOR_R_64F);
    typedef double floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.0;
    floatTypeCompute beta = (floatTypeCompute)0.;

    // use default algorithm for contraction
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref,
                                               &workspaceSizeEstimate));

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                          &actualWorkspaceSize, sizeof(actualWorkspaceSize)));

    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0);
    }

    // create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // perform the cuTENSOR contraction
    HANDLE_ERROR(cutensorContract(handle, plan, (void *)&alpha, A_d, B_d, (void *)&beta, C_d, C_d,
                                  work, actualWorkspaceSize, stream));

    // copy resulting tensor contraction data from GPU back to CPU
    CUDA_CHECK(cudaMemcpy(C_data.data(), C_d, sizeC, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // free the data
    if (A_d)
        cudaFree(A_d);
    if (B_d)
        cudaFree(B_d);
    if (C_d)
        cudaFree(C_d);
    if (work)
        cudaFree(work);

    return Tensor(indicesC, C_data);
}

Tensor &Tensor::operator*=(const Tensor &other) {
    std::vector<std::vector<int>> contract_modes = find_contract_modes(indices, other.indices);

    // A refers to `this`, B refers to `other`, and C refers to the resulting tensor from the
    // multiplication

    // number of modes
    std::vector<int> modeA = contract_modes[0];
    std::vector<int> modeB = contract_modes[1];
    std::vector<int> modeC = contract_modes[2];
    std::vector<int> posC = contract_modes[3];
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // create the vectors of extents for each tensor and the number of elements
    std::vector<int64_t> extentA(nmodeA);
    size_t elementsA = 1;
    for (int i = 0; i < nmodeA; i++) {
        extentA[i] = indices[i].get_dims();
        elementsA *= extentA[i];
    }
    size_t elementsB = 1;
    std::vector<int64_t> extentB(nmodeB);
    for (int i = 0; i < nmodeB; i++) {
        extentB[i] = other.indices[i].get_dims();
        elementsB *= extentB[i];
    }
    size_t elementsC = 1;
    std::vector<int64_t> extentC(nmodeC);
    std::vector<size_t> shapeC(nmodeC);
    std::vector<Index> indicesC(nmodeC);
    for (int i = 0; i < nmodeC; i++) {
        int pos = posC[i];
        if (pos < nmodeA) {
            extentC[i] = indices[pos].get_dims();
            indicesC[i] = indices[pos];
        } else {
            extentC[i] = other.indices[pos - nmodeA].get_dims();
            indicesC[i] = other.indices[pos - nmodeA];
        }
        shapeC[i] = extentC[i];
        elementsC *= extentC[i];
    }

    // get size in bytes
    size_t sizeA = sizeof(double) * elementsA;
    size_t sizeB = sizeof(double) * elementsB;
    size_t sizeC = sizeof(double) * elementsC;

    // allocate on device
    void *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, sizeA);
    cudaMalloc((void **)&B_d, sizeB);
    cudaMalloc((void **)&C_d, sizeC);

    // make empty xarray for C
    xt::xarray<double, xt::layout_type::column_major> C_data = xt::zeros<double>(shapeC);

    // copy to device
    CUDA_CHECK(cudaMemcpy(A_d, data.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, (other.data).data(), sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_data.data(), sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    // perform cuTENSOR's tensor contraction
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    // create the tensor descriptions for A, B, and C
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, nmodeA, extentA.data(), NULL,
                                                CUTENSOR_R_64F, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, nmodeB, extentB.data(), NULL,
                                                CUTENSOR_R_64F, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, nmodeC, extentC.data(), NULL,
                                                CUTENSOR_R_64F, kAlignment));

    // create the tensor description for the contraction
    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                                           descB, modeB.data(), CUTENSOR_OP_IDENTITY, descC,
                                           modeC.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(),
                                           CUTENSOR_COMPUTE_DESC_64F));

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void *)&scalarType, sizeof(scalarType)));

    // use double
    assert(scalarType == CUTENSOR_R_64F);
    typedef double floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.0;
    floatTypeCompute beta = (floatTypeCompute)0.;

    // use default algorithm for contraction
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref,
                                               &workspaceSizeEstimate));

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                          &actualWorkspaceSize, sizeof(actualWorkspaceSize)));

    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0);
    }

    // create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // perform the cuTENSOR contraction
    HANDLE_ERROR(cutensorContract(handle, plan, (void *)&alpha, A_d, B_d, (void *)&beta, C_d, C_d,
                                  work, actualWorkspaceSize, stream));

    // copy resulting tensor contraction data from GPU back to CPU
    CUDA_CHECK(cudaMemcpy(C_data.data(), C_d, sizeC, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // free the data
    if (A_d)
        cudaFree(A_d);
    if (B_d)
        cudaFree(B_d);
    if (C_d)
        cudaFree(C_d);
    if (work)
        cudaFree(work);

    indices = indicesC;
    num_indices = indices.size();
    data = C_data;

    return *this;
}

std::tuple<Tensor, Tensor, xt::xarray<double, xt::layout_type::column_major>, int, double>
Tensor::svd(std::vector<int> &a, std::vector<int> &b, bool mergeSV, double cutoff, int mindim,
            int maxdim) {

    // create CUDA handle and stream
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    // get the number of rows and columns
    std::size_t a_size = 1;
    std::size_t b_size = 1;

    for (int i = 0; i < a.size(); i++) {
        a_size *= indices[a[i]].get_dims();
    }

    for (int j = 0; j < b.size(); j++) {
        b_size *= indices[b[j]].get_dims();
    }

    int m;
    int n;

    // gsevd only supports m >= n
    // if b_size < a_size, we use the tranposed M then transpose the results back
    if (a_size >= b_size) {
        m = a_size;
        n = b_size;
    } else {
        m = b_size;
        n = a_size;
    }
    const int lda = m;
    const int ldu = m;
    const int ldvt = n;

    // copy the tensor data to `M_data`
    xt::xarray<double, xt::layout_type::column_major> M_data = data;
    M_data.reshape({a_size, b_size});

    // transpose to ensure that m >= n
    if (a_size < b_size) {
        M_data = xt::transpose(M_data);
    }

    // initialize `xt::xarray` objects to hold the singular value decomposition results
    xt::xarray<double, xt::layout_type::column_major> U_data = xt::zeros<double>({m, m});
    xt::xarray<double, xt::layout_type::column_major> S_data = xt::zeros<double>({std::min(m, n)});
    xt::xarray<double, xt::layout_type::column_major> Vt_data = xt::zeros<double>({n, n});

    // get size in bytes
    size_t sizeM = sizeof(double) * m * n;
    size_t sizeU = sizeof(double) * m * m;
    size_t sizeS = sizeof(double) * std::min(m, n);
    size_t sizeVt = sizeof(double) * n * n;

    double *M_d = nullptr;
    double *U_d = nullptr;
    double *S_d = nullptr;
    double *Vt_d = nullptr;

    int *devInfo = nullptr;

    int lwork = 0;
    double *d_work = nullptr;
    double *rwork = nullptr;

    // create the cusolver handle and bind a stream
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    // allocate on GPU
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&M_d), sizeM));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&U_d), sizeU));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&S_d), sizeS));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Vt_d), sizeVt));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    // copy data to GPU
    CUDA_CHECK(cudaMemcpyAsync(M_d, M_data.data(), sizeM, cudaMemcpyHostToDevice, stream));

    // create workspace size and allocate on GPU
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), lwork * sizeof(double)));

    // perform SVD
    signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n rows of VT
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, M_d, lda, S_d, U_d, ldu, Vt_d,
                                    ldvt, d_work, lwork, rwork, devInfo));

    // check if the computation was successful
    int info;
    CUDA_CHECK(cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (info != 0) {
        std::cerr << "Error in " << __FILE__ << ": SVD failed: " << info << std::endl;
    }

    // copy results back to the CPU
    CUDA_CHECK(cudaMemcpyAsync(U_data.data(), U_d, sizeU, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(S_data.data(), S_d, sizeS, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(Vt_data.data(), Vt_d, sizeVt, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // free up resources
    cudaFree(M_d);
    cudaFree(S_d);
    cudaFree(U_d);
    cudaFree(Vt_d);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);

    // transpose back if a_size < b_size
    if (a_size < b_size) {
        M_data = xt::transpose(M_data);
        xt::xarray<double, xt::layout_type::column_major> U_temp = xt::transpose(Vt_data);
        xt::xarray<double, xt::layout_type::column_major> Vt_temp = xt::transpose(U_data);

        // get transposed data
        U_data = U_temp;
        Vt_data = Vt_temp;
    }

    // perform truncation using ITensor.jl's truncation logic: see
    // `ITensors.jl/NDTensors/src/truncate.jl`
    int s_dim = std::min(m, n); // actual dim after truncation
    xt::xarray<double, xt::layout_type::column_major> P_data = S_data * S_data;

    // zero out any negative weights
    for (int i = s_dim - 1; i >= 0; i--) {
        if (P_data[i] >= (double)0.0) {
            break;
        } else {
            P_data[i] = (double)0.0;
        }
    }

    // initially truncate with `maxdim` parameter
    double trunc_err = (double)0.0;
    while (s_dim > maxdim) {
        trunc_err += P_data[s_dim - 1];
        s_dim -= 1;
    }

    // now truncate using `cutoff` parameter
    double scale;
    scale = xt::sum(P_data)();
    if (scale == (double)0.0) {
        scale = (double)1.0;
    }

    // continue truncating until trunc_error reaches cutoff * scale
    while (trunc_err + P_data[s_dim - 1] <= cutoff * scale && s_dim > mindim) {
        trunc_err += P_data[s_dim - 1];
        s_dim -= 1;
    }

    trunc_err /= scale;

    // get truncated singular value decomposition tensors
    U_data = xt::eval(xt::view(U_data, xt::all(), xt::range(0, s_dim)));
    Vt_data = xt::eval(xt::view(Vt_data, xt::range(0, s_dim), xt::all()));
    S_data = xt::eval(xt::view(S_data, xt::range(0, s_dim)));

    if (mergeSV) {
        Vt_data = xt::linalg::dot(xt::diag(S_data), Vt_data); // merge SV^T
    } else {
        U_data = xt::linalg::dot(U_data, xt::diag(S_data)); // merge US
    }

    // `Index` object for `S`
    Index mid_idx(s_dim);

    std::vector<Index> U_indices(a.size() + 1);
    std::vector<Index> Vt_indices(b.size() + 1);

    for (int i = 0; i < a.size(); i++) {
        U_indices[i] = indices[a[i]];
    }
    U_indices[a.size()] = mid_idx;

    for (int j = 0; j < b.size(); j++) {
        Vt_indices[j + 1] = indices[b[j]];
    }
    Vt_indices[0] = mid_idx;

    std::vector<size_t> U_shape(U_indices.size());
    for (int i = 0; i < U_indices.size(); i++) {
        U_shape[i] = U_indices[i].get_dims();
    }

    Tensor U_tensor = Tensor(U_indices, U_data.reshape(U_shape));

    std::vector<size_t> Vt_shape(Vt_indices.size());
    for (int i = 0; i < Vt_indices.size(); i++) {
        Vt_shape[i] = Vt_indices[i].get_dims();
    }

    Tensor Vt_tensor = Tensor(Vt_indices, Vt_data.reshape(Vt_shape));

    return std::make_tuple(U_tensor, Vt_tensor, S_data, s_dim, trunc_err);
}
