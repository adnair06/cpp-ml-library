#pragma once

#include "Matrix.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

struct Node {
    bool leaf; 

    int feature; 
    double threshold; 

    int prediction; 

    Node* left; 
    Node* right;
};

class DecisionTreeClassifier {
private: 
    Node* root; 

public:  
    double entropy (const Matrix& y) const {
        double result = 0; 
        size_t count_0 = 0, count_1 = 0; 
        size_t total = y.rows();
        for (size_t i = 0; i < total; i++) {
            if (y(i, 0) == 0) {
                count_0++;  
            } else if (y(i, 0) == 1) {
                count_1++;  
            } else {
                throw std::invalid_argument("Matrix must have labels 0 or 1 only."); 
            }
        }

        
        double p_0 = count_0/total, p_1 = count_1/total; 
        if (p_0 > 0) {
            result -= p_0 * log2(p_0); 
        }

        if (p_1 > 0) {
            result -= p_1 * log2(p_1); 
        }

        return result; 
    }

    void fit(const Matrix& X, const Matrix& y); 

    int predict_row(const Matrix& X, size_t row, const Node* node) const {
        if (node -> leaf) {
            return node -> prediction; 
        }

        if (X(row, node -> feature) <= node -> threshold) {
            return predict_row(X, row, node -> left); 
        }

        return predict_row(X, row, node -> right); 
    }

    Matrix predict(const Matrix& X) const {
        if (root == nullptr) {
            throw std::invalid_argument("Decision tree has not been fitted."); 
        }

        Matrix result(X.rows(), 1); 
        for (size_t i = 0; i < X.rows(); i++) {
            result(i, 0) = predict_row(X, i, root); 
        }

        return result; 
    } 
};