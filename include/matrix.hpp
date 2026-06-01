#pragma once

#include <vector>
#include <stdexcept>
#include <cstddef>

class Matrix {
private:
    std::vector<double> data;
    size_t n_rows;
    size_t n_cols;

public:
    Matrix() : n_rows(0), n_cols(0) {}

    Matrix (size_t rows, size_t cols) : data(rows*cols), n_rows(rows), n_cols(cols) {}

    size_t rows() const {
        return n_rows;
    }

    size_t cols() const {
        return n_cols;
    }

    double operator()(size_t i, size_t j) const {
        if (i >= n_rows || j >= n_cols) {
            throw std::invalid_argument("Matrix indexing out of bounds."); 
        }
        return data[i*n_cols + j];
    }

    double& operator()(size_t i, size_t j) {
        if (i >= n_rows || j >= n_cols) {
            throw std::invalid_argument("Matrix indexing out of bounds.");
        }

        return data[i * n_cols + j];
    }

    Matrix operator+(const Matrix& other) const { //Revise the operator overloading here.
        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            throw std::invalid_argument("Matrix sizes do not match."); 
        }

        Matrix result(n_rows, n_cols); 
        for (size_t i = 0; i < n_rows*n_cols; i++) {
            result.data[i] = this->data[i] + other.data[i]; 
        }

        return result;
    }

    Matrix operator-(const Matrix& other) const { //Revise the operator overloading here.
        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            throw std::invalid_argument("Matrix sizes do not match."); 
        }

        Matrix result(n_rows, n_cols); 
        for (size_t i = 0; i < n_rows*n_cols; i++) {
            result.data[i] = this->data[i] - other.data[i]; 
        }

        return result;
    }

    Matrix operator*(const double& other) const {
        Matrix result(n_rows, n_cols); 
        for (size_t i = 0; i < n_rows*n_cols; i++) {
            result.data[i] = other*this->data[i];
        } 

        return result;
    }

    Matrix transpose() const {
        Matrix result(n_cols, n_rows); 
        for (size_t i = 0; i < n_rows; i++) {
            for (size_t j = 0; j < n_cols; j++) {
                result.data[j*n_rows + i] = data[i*n_cols + j];
            }
        }

        return result; 
    }

    Matrix operator*(const Matrix& other) const {
        if (this -> n_cols != other.n_rows) {
            throw std::invalid_argument("Matrix dimensions are not valid.");
        }

        Matrix result(n_rows, other.n_cols); 
        for (size_t i = 0; i < n_rows; i++) {
            for (size_t j = 0; j < other.n_cols; j++) {
                double sum = 0; 
                for (size_t k = 0; k < this -> n_cols; k++) {
                    sum += (this -> data[i*n_cols + k])*(other.data[k*other.n_cols + j]);
                }
                result.data[i*other.n_cols + j] = sum; 
            }
        }

        return result;
    }

    Matrix add_bias_column() const {
        Matrix result(n_rows, n_cols + 1);

        for (size_t i = 0; i < n_rows; i++) {

            result(i, 0) = 1.0;

            for (size_t j = 0; j < n_cols; j++) {
                result(i, j + 1) = data[i * n_cols + j];
            }
        }

        return result;
    }

    Matrix slice(size_t start, size_t end) const {
        if (start >= end || end > n_rows) {
            throw std::invalid_argument("Input parameters are out of range."); 
        }

        Matrix result(end - start, n_cols); 
        for (size_t i = 0; i < end - start; i++) {
            for (size_t j = 0; j < n_cols; j++) {
                result(i, j) = data[(start + i)*n_cols + j]; 
            }
        }

        return result; 
    }
};