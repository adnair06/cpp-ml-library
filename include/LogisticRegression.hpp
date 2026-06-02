#pragma once

#include "Matrix.hpp"
#include <cmath>

class LogisticRegression {
private:
    Matrix W; 
public:
    Matrix predict(const Matrix& X) const {
        Matrix X_bias = X; 
        if (X.cols() == W.rows() - 1) {
            X_bias = X.add_bias_column(); 
        }
        
        Matrix mult = X_bias * W; 
        Matrix result(mult.rows(), mult.cols()); 
        for (size_t i = 0; i < mult.rows(); i++) {
            for (size_t j = 0; j < mult.cols(); j++) {
                result(i, j) = 1.0 / (1.0 + std::exp(-1 * mult(i, j)));
            }
        }

        return result; 
    }

    void fit(const Matrix& X, const Matrix& y, double learning_rate, size_t epochs) {
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("Dimensions of the input(X) and output(y) doesn't match."); 
        }

        size_t n = X.cols(), m = X.rows(); 
        Matrix X_bias = X.add_bias_column(); 
        W = Matrix(n + 1, 1); 
        Matrix XT = X_bias.transpose(); 
        size_t iterations = 0;
        Matrix diff, mult, pred; 
        while (iterations++ != epochs) {
            pred = X_bias * W;

            for (size_t i = 0; i < m; i++) {
                pred(i,0) = 1.0 / (1.0 + std::exp(-pred(i,0)));
            }

            diff = y - pred; 
            mult = XT * (diff); 
            W = W + (mult) * (learning_rate / static_cast<double>(m));
        }
    }

    static double cross_entropy_loss(const Matrix& y, const Matrix& y_pred) {
        if (y.rows() != y_pred.rows() || y.cols() != 1 || y_pred.cols() != 1) {
            throw std::invalid_argument("Matrix dimensions do not match"); 
        }

        double loss = 0; 
        for (size_t i = 0; i < y.rows(); i++) {
            double p = y_pred(i, 0); 
            p = std::max(p, 1e-15); 
            p = std::min(p, 1.0 - 1e-15);
            loss += -1 * (y(i, 0) * std::log(p) + (1 - y(i, 0)) * std::log(1 - p)); 
        }

        return loss / static_cast<double>(y.rows()); 
    }
};