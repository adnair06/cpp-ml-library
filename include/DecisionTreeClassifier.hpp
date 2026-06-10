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

    DecisionTreeClassifier(size_t max_depth = 10, size_t min_samples_split = 2) : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split) {}

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

        
        double p_0 = static_cast<double>(count_0)/static_cast<double>(total), p_1 = static_cast<double>(count_1)/static_cast<double>(total); 
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

    int majority_class(const Matrix& y) const {
        size_t cnt = 0; 
        for (size_t i = 0; i < y.rows(); i++) {
            if (y(i, 0)) {
                cnt++; 
            }
        }

        return ((cnt >= y.rows() - cnt) ? 1 : 0);  
    } 

    Node* build_tree(const Matrix& X, const Matrix& y, size_t depth) {
        Node* currnode = new Node(); 
        bool allequal = true; 

        if (y.rows() == 0) {
            throw std::invalid_argument("Empty dataset.");
        }

        for (size_t i = 0; i < y.rows() - 1; i++) {
            if (y(i, 0) == y(i + 1, 0)) {
                continue;
            } else {
                allequal = false; 
                break; 
            }
        }

        if (allequal || X.rows() < min_samples_split || depth >= max_depth) {
            currnode -> leaf = true; 
            currnode -> right = nullptr; 
            currnode -> left = nullptr; 
            size_t cnt = 0;  
            for (size_t i = 0; i < y.rows(); i++) {
                if (y(i, 0)) {
                    cnt++; 
                }
            }

            currnode -> prediction = majority_class(y);
            return currnode; 
        }

        Split best = find_best_split(X, y); 
        if (best.feature == -1) {
            currnode -> leaf = true; 
            currnode -> prediction = majority_class(y); 
            currnode -> right = nullptr; 
            currnode -> left = nullptr; 

            return currnode; 
        } else {
            currnode -> leaf = false; 
            currnode -> feature = best.feature; 
            currnode -> threshold = best.threshold;
        }

        size_t rcount = 0; 
        for (size_t i = 0; i < X.rows(); i++) {
            if (X(i, best.feature) > best.threshold) {
                rcount++; 
            }
        }

        Matrix X_r(rcount, X.cols()), X_l(X.rows() - rcount, X.cols()); 
        Matrix y_r(rcount, 1), y_l(X.rows() - rcount, 1); 
        size_t last_r = 0, last_l = 0; 
        for (size_t i = 0; i < X.rows(); i++) {
            if (X(i, best.feature) > best.threshold) {
                for (size_t j = 0; j < X.cols(); j++) {
                    X_r(last_r, j) = X(i, j);  
                }
                y_r(last_r, 0) = y(i, 0); 

                last_r++;
            } else {
                for (size_t j = 0; j < X.cols(); j++) {
                    X_l(last_l, j) = X(i, j); 
                }
                y_l(last_l, 0) = y(i, 0); 

                last_l++; 
            }
        }

        currnode -> right = build_tree(X_r, y_r, depth + 1); 
        currnode -> left = build_tree(X_l, y_l, depth + 1); 

        return currnode; 
    }

    void fit(const Matrix& X, const Matrix& y) {
        root = build_tree(X, y, 0);
    }

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