#pragma once

#include "matrix.hpp"
#include "DataFrame.hpp"

#include <cstddef>
#include <stdexcept>
#include <cmath>

class LinearRegression {
private:
    Matrix W; 
public:
    void fit(const Matrix& X, const Matrix& y, double learning_rate, size_t epochs) {
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("Dimensions of the input(X) and output(y) doesn't match."); 
        }

        size_t n = X.cols(), m = X.rows(); 
        Matrix X_bias = X.add_bias_column(); 
        W = Matrix(n + 1, 1); 
        size_t iterations = 0; 
        while(iterations++ != epochs) {
            W = W - (X_bias.transpose() * (X_bias*W - y)) * (learning_rate/static_cast<double>(n)); 
        }
    }

    Matrix predict (const Matrix& X) const {
        Matrix X_bias = X.add_bias_column(); 
        return X_bias * W;  
    }

    static double MSE_Error(const Matrix& y_test, const Matrix& y_predicted) {

        if (y_test.rows() != y_predicted.rows() || y_test.cols() != y_predicted.cols()) {
            throw std::invalid_argument("Dimensions of actual and predicted outputs do not match.");
        }

        size_t n = y_test.rows();
        Matrix e = y_test - y_predicted;
        Matrix mult = e.transpose() * e;
        return mult(0,0) / static_cast<double>(n);
    }

    static double MAPE_Error(const Matrix& y_test, const Matrix& y_predicted) {

        if (y_test.rows() != y_predicted.rows() || y_test.cols() != y_predicted.cols()) {
            throw std::invalid_argument("Dimensions of actual and predicted outputs do not match.");
        }

        size_t n = y_test.rows();
        double sum = 0;

        for (size_t i = 0; i < n; i++) {
            double actual = y_test(i, 0);
            double predicted = y_predicted(i, 0);
            if (actual == 0) {
                throw std::invalid_argument("MAPE undefined when actual value is zero.");
            }
            sum += std::abs((actual - predicted) / actual);
        }

        return (100.0 * sum) / n;
    }
};

class Scaler {
private:
    std::vector<double> mean; 
    std::vector<double> std; 

public:
    void fit(const Matrix& X) {
        size_t rows = X.rows(), cols = X.cols(); 
        if (rows == 0) {
            throw std::invalid_argument("Empty matrix cannot be fitted."); 
        }
        mean.resize(cols); 
        std.resize(cols); 
        for (size_t i = 0; i < cols; i++) {
            for (size_t j = 0; j < rows; j++) {
                mean[i] += X(j, i); 
                std[i] += X(j, i) * X(j, i); 
            }

            mean[i] /= rows;
            std[i] /= rows;
            std[i] = std::sqrt(std[i] - mean[i] * mean[i]);
        }
    }

    Matrix transform(const Matrix& X) const {
        size_t rows = X.rows(), cols = X.cols(); 
        if (cols != mean.size()) {
            throw std::invalid_argument("Input matrix dimensions does not match fitted matrix.");
        }
        Matrix result(rows, cols); 

        for (size_t i = 0; i < cols; i++) {
            if (std[i] == 0) {
                throw std::invalid_argument("Cannot scale constant feature."); 
            }
            for (size_t j = 0; j < rows; j++) {
                result(j, i) = (X(j, i) - mean[i])/std[i]; 
            }
        }

        return result; 
    }

    Matrix inverse_transform(const Matrix& X) const {
        Matrix result(X.rows(), X.cols());

        for (size_t i = 0; i < X.cols(); i++) {
            for (size_t j = 0; j < X.rows(); j++) {
                result(j,i) = X(j,i) * std[i] + mean[i];
            }
        }

        return result;
    }
};