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

struct Split {
    int feature; 
    double threshold; 
    double gain; 
};

class DecisionTreeClassifier {
private: 
    Node* root; 

    size_t max_depth, min_samples_split; 
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

    double information_gain(const Matrix& y, const Matrix& left, const Matrix& right) const {
        if (left.rows() == 0 || right.rows() == 0) {
            return -std::numeric_limits<double>::infinity(); 
        }
        double y_entropy = entropy(y); 
        double right_entropy = entropy(right); 
        double left_entropy = entropy(left); 

        double total = static_cast<double>(y.rows()); 
        double right_w = static_cast<double>(right.rows()) / total; 
        double left_w = static_cast<double>(left.rows()) / total; 

        return (y_entropy - (right_w * right_entropy + left_w * left_entropy));  
    }

    Split find_best_split(const Matrix& X, const Matrix& y) const {
        Split best_split; 
        best_split.feature = -1;
        best_split.threshold = 0;
        best_split.gain = -std::numeric_limits<double>::infinity();
        std::vector<double> currcol, right, left; 
        Matrix left_m, right_m; 

        size_t n = X.rows(), m = X.cols(); 
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                currcol.push_back(X(j, i)); 
            }

            sort(currcol.begin(), currcol.end()); 

            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < n; k++) {
                    if (X(k, i) <= currcol[j]) {
                        left.push_back(y(k, 0));
                    } else {
                        right.push_back(y(k, 0)); 
                    }
                }

                left_m.resize(left.size(), 1), right_m.resize(right.size(), 1); 
                for (size_t k = 0; k < left.size(); k++) {
                    left_m(k, 0) = left[k]; 
                }
                for (size_t k = 0; k < right.size(); k++) {
                    right_m(k, 0) = right[k]; 
                }

                double curr_gain = information_gain(y, left_m, right_m);
                
                if (curr_gain > best_split.gain) {
                    best_split.gain = curr_gain; 
                    best_split.feature = i; 
                    best_split.threshold = currcol[j]; 
                }

                left.clear(); 
                right.clear(); 
            }

            currcol.clear(); 
        }

        return best_split; 
    }

    Node* build_tree(const Matrix& X, const Matrix& y, size_t depth) {
        Node* currnode; 
        bool allequal = true; 
        for (size_t i = 0; i < y.rows() - 1; i++) {
            if (y(i, 0) == y(i + 1, 0)) {
                continue;
            } else {
                allequal = false; 
                break; 
            }
        }

        if (allequal || X.rows() <= min_samples_split || depth >= max_depth) {
            currnode -> leaf = true; 
            size_t cnt = 0;  
            for (size_t i = 0; i < y.rows(); i++) {
                if (y(i, 0)) {
                    cnt++; 
                }
            }

            if (cnt >= y.rows() - cnt) {
                currnode -> prediction = 1; 
            } else {
                currnode -> prediction = 0; 
            }

            return currnode; 
        }

        Split best = find_best_split(X, y); 
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