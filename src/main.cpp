#include <iostream> 
#include <algorithm> 
#include <vector>
#include "DataFrame.hpp"
#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include "DecisionTreeClassifier.hpp"

int main() {
    DataFrame df = CSVReader::read("data/Diabetes_Prediction.csv"); 

    df.shuffle(); 
    df.one_hot_encode("gender"); 

    std::vector<std::string> header = df.headers(); 
    header.erase(std::find(header.begin(), header.end(), "smoking_history")); 
    header.erase(std::find(header.begin(), header.end(), "diabetes"));

    Matrix X = df.to_matrix(header), y = df.to_matrix({"diabetes"}); 
    Matrix X_train = X.slice(0, 8000), X_test = X.slice(8000, 10000), y_train = y.slice(0, 8000), y_test = y.slice(8000, 10000); 

    // Scaler Xs; 
    // Xs.fit(X_train); 
    // Matrix X_train_Scaled = Xs.transform(X_train), X_test_Scaled = Xs.transform(X_test); 

    DecisionTreeClassifier model; 
    model.fit(X_train, y_train);
    Matrix y_pred = model.predict(X_test); 
    
    double acc = LogisticRegression::accuracy(y_test, y_pred); 
    std::cout << "Decision tree classifier has an accuracy of: " << acc << std::endl;
}