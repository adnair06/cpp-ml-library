#include <iostream> 
#include <algorithm> 
#include <vector>
#include "DataFrame.hpp"
#include "LinearRegression.hpp"

int main() {
    DataFrame df = CSVReader::read("data/Housing.csv"); 
    df.shuffle(); 
    std::vector<std::string> categorical = {"mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"};
    for (auto& h: categorical) {
        df.one_hot_encode(h); 
    }
    std::vector<std::string> columns = df.headers(); 
    columns.erase(std::find(columns.begin(), columns.end(), "price"));  

    Matrix X = df.to_matrix(columns); 
    Matrix y = df.to_matrix({"price"}); 

    Matrix X_train = X.slice(0, 400), X_test = X.slice(400, X.rows()), y_train = y.slice(0, 400), y_test = y.slice(400, y.rows()); 
    Scaler Xs, ys; 
    Xs.fit(X_train); 
    ys.fit(y_train); 
    Matrix X_train_scaled = Xs.transform(X_train), y_train_scaled = ys.transform(y_train), X_test_scaled = Xs.transform(X_test), y_test_scaled = ys.transform(y_test); 

    LinearRegression model; 
    model.fit(X_train_scaled, y_train_scaled, 0.01, 10000); 
    Matrix predictions_scaled = model.predict(X_test_scaled); 
    Matrix predictions = ys.inverse_transform(predictions_scaled);

    double error = LinearRegression::MAPE_Error(y_test, predictions); 
    std::cout << "Error in Linear Regression: " << error << "%" << std::endl;
    return 0; 
}