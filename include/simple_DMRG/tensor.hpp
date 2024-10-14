/**
 * @file tensor.hpp
 * @brief Defines the `Tensor` class that uses `Index` to represent a single tensor object.
 *
 *
 * Author: Hyeongjin Kim
 * Date: 2024
 * Version: 1.0
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <tuple>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xrandom.hpp>

// cutensor libraries
#include <cuda_runtime.h>
#include <cutensor.h>

// cuda libraries
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "index.hpp"

/**
 * @brief A class that represents a multi-dimension tensor.
 *
 * The `Tensor` class contains the tensor data and the associated indices (a collection of `Index`
 * objects). This class allows for various tensor operations such as scalar multiplication,
 * multiplication between tensors, and singular value decomposition of tensors.
 */
class Tensor {
public:
    /**
     * @brief Default constructor for `Tensor`.
     *
     * This constructor initializes a default `Tensor` object with zeros.
     */
    Tensor();

    /**
     * @brief Constructor for `Tensor`.
     *
     * This constructor initializes a `Tensor` object with a user-specified collection of `Index`
     * objects from `indices`. The tensor data is initialized to zeros.
     *
     * @param indices The collection of `Index` objects: each index represents a dimension.
     */
    Tensor(std::vector<Index> indices);

    /**
     * @brief Constructor for `Tensor`.
     *
     * This constructor initializes a `Tensor` object with a user-specified collection of `Index`
     * objects from `indices`. The tensor data is initialized using a specified `xt::xarray` object.
     *
     * @param indices The collection of `Index` objects: each index represents a dimension.
     * @param data The `xt::xarray` object to initialize the tensor data: shape must match!
     */
    Tensor(std::vector<Index> indices,
           const xt::xarray<double, xt::layout_type::column_major> &set_data);

    /**
     * @brief Default deconstructor for `Tensor`.
     */
    ~Tensor();

    /**
     * @brief Copy constructor for `Tensor`.
     *
     * @param other The `Tensor` object to copy from.
     */
    Tensor(const Tensor &other);

    /**
     * @brief Assignment operator for `Tensor`.
     *
     * @param other The `Tensor` object to assign from.
     * @return A reference to this `Tensor` object.
     */
    Tensor &operator=(const Tensor &other);

    /**
     * @brief Sets all elements of the tensor to one.
     *
     * @return A reference to this `Tensor` object.
     */
    Tensor &set_one();

    /**
     * @brief Sets all elements of the tensor to zero.
     *
     * @return A reference to this `Tensor` object.
     */
    Tensor &set_zero();

    /**
     * @brief Randomizes all elements of the tensor.
     *
     * @return A reference to this `Tensor` object.
     */
    Tensor &set_random();

    /**
     * @brief Gets the tensor data.
     *
     * @return The tensor data as an `xt::xarray`.
     */
    xt::xarray<double, xt::layout_type::column_major> get_data() const;

    /**
     * @brief Gets the collection of indices.
     *
     * @return The collection of `Index` objects as an `std::vector`.
     */
    std::vector<Index> get_indices() const;

    /**
     * @brief Gets the number of `Index` objects
     *
     * @return The number of `Index` objects, or the number of dimensions of tensor.
     */
    int get_num_indices() const;

    /**
     * @brief Sets the tensor data.
     *
     * @param new_data The new tensor data to set.
     * @return A reference to this `Tensor` object with the new tensor data.
     */
    Tensor &set_data(const xt::xarray<double, xt::layout_type::column_major> &new_data);

    /**
     * @brief Raises the prime levels of all the indices.
     *
     * @return A reference to this `Tensor` object.
     */
    Tensor &prime_indices();

    /**
     * @brief Lowers the prime levels of all the indices.
     *
     * @return A reference to this `Tensor` object.
     */
    Tensor &unprime_indices();

    /**
     * @brief Normalizes the tensor data.
     *
     * @return A reference to this `Tensor` object.
     */
    Tensor &normalize();

    /**
     * @brief Get the L2 norm of the `Tensor` object.
     *
     * @return The L2 norm of the tensor data.
     */
    double get_norm() const;

    /**
     * @brief Adds a scalar to the tensor.
     *
     * @param scalar The scalar value to add.
     * @return A new `Tensor` object resulting from the scalar addition.
     */
    Tensor operator+(const double &scalar) const;

    /**
     * @brief Adds a scalar to the tensor.
     *
     * @param scalar The scalar value to add.
     * @return A reference to this `Tensor` object after scalar addition.
     */
    Tensor &operator+=(const double &scalar);

    /**
     * @brief Subtracts a scalar from the tensor.
     *
     * @param scalar The scalar value to subtract.
     * @return A new `Tensor` object resulting from the scalar subtraction.
     */
    Tensor operator-(const double &scalar) const;

    /**
     * @brief Subtracts a scalar from the tensor.
     *
     * @param scalar The scalar value to subtract.
     * @return A reference to this `Tensor` object after scalar subtraction.
     */
    Tensor &operator-=(const double &scalar);

    /**
     * @brief Multiplies a scalar to the tensor.
     *
     * @param scalar The scalar value to multiply.
     * @return A new `Tensor` object resulting from the scalar multiplication.
     */
    Tensor operator*(const double &scalar) const;

    /**
     * @brief Multiplies a scalar to the tensor.
     *
     * @param scalar The scalar value to multiply.
     * @return A reference to this `Tensor` object after scalar multiplication.
     */
    Tensor &operator*=(const double &scalar);

    /**
     * @brief Adds a tensor to the tensor.
     *
     * @param other The `Tensor` object to add.
     * @return A new `Tensor` object after the tensor addition.
     */
    Tensor operator+(const Tensor &other) const;

    /**
     * @brief Adds a tensor to the tensor.
     *
     * @param other The `Tensor` object to add.
     * @return A reference to this `Tensor` object after tensor addition.
     */
    Tensor &operator+=(const Tensor &other);

    /**
     * @brief Subtracts a tensor from the current tensor.
     *
     * @param other The `Tensor` object to subtract.
     * @return A new `Tensor` object after the tensor subtraction.
     */
    Tensor operator-(const Tensor &other) const;

    /**
     * @brief Subtracts a tensor from the current tensor.
     *
     * @param other The `Tensor` object to subtract.
     * @return A reference to this `Tensor` object after tensor subtraction.
     */
    Tensor &operator-=(const Tensor &other);

    /**
     * @brief Multiplies a tensor to the current tensor.
     *
     * @param other The `Tensor` object to multiply.
     * @return A new `Tensor` object after the tensor multiplication.
     */
    Tensor operator*(const Tensor &other) const;

    /**
     * @brief Multiplies a tensor to the current tensor.
     *
     * @param other The `Tensor` object to multiply.
     * @return A reference to this `Tensor` object after tensor multiplication.
     */
    Tensor &operator*=(const Tensor &other);

    /**
     * @brief Performs the singular value decomposition (SVD) of the tensor.
     *
     * This function merges the indices `a` as the rows and indices `b` as the columns to transform
     * the `Tensor` object into a two-dimensional matrix. Then, this method decomposes the tensor T
     * into USV^T format and either merges US -> U if `mergeSV = false` or SV^T -> V^T if `mergeSV =
     * true`. The resulting decompositions are given as `Tensor` objects.
     *
     * @param a The row indices.
     * @param b The column indices.
     * @param mergeSV A boolean flag indicating whether to merge U with S or S with V^T.
     * @param cutoff The truncation cutoff value for singular values (default is 1e-10).
     * @param mindim The minimum number of singular values to hold (default is 1).
     * @param maxdim The maximum number of singular values to hold (default is 100).
     * @return A tuple containing the resulting tensors, the singular values, the number of
     * dimensions, and the truncation error.
     */
    std::tuple<Tensor, Tensor, xt::xarray<double, xt::layout_type::column_major>, int, double>
    svd(std::vector<int> &a, std::vector<int> &b, bool mergeSV = true, double cutoff = 1e-10,
        int mindim = 1, int maxdim = 100);

private:
    /**
     * @brief The vector of `Index` objects.
     */
    std::vector<Index> indices;

    /**
     * @brief The number of `Index` objects / dimension of the `Tensor` object.
     */
    int num_indices;

    /**
     * @brief The tensor data.
     */
    xt::xarray<double, xt::layout_type::column_major> data;
};

#endif // TENSOR_HPP
