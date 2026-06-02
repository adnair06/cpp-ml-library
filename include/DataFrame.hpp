#pragma once 

#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include<set>
#include "Matrix.hpp"

enum class DataType {
    INT,
    DOUBLE, 
    STRING
};

class Column {
private: 
    std::variant<
        std::vector<int>, 
        std::vector<double>, 
        std::vector<std::string>
    > data; //The actual column that stores all the data. 
public: 
    Column() = default; //Constructor without initialization of the datatype. -> Default is int. 

    Column(DataType type){ //Constructor for initialization with declaration of datatype.
        if (type == DataType::INT) {
            data = std::vector<int>();
        } else if (type == DataType::DOUBLE) {
            data = std::vector<double>(); 
        } else if (type == DataType::STRING) {
            data = std::vector<std::string>(); 
        } else {
            throw std::invalid_argument("Invalid DataType");
        }
    } 

    DataType type() const { //Const added here because then this function promises to not modify the object. So this function can also be called when a const Column in declared.
        if (std::holds_alternative<std::vector<int>>(data)) {
            return DataType::INT; 
        } else if (std::holds_alternative<std::vector<double>>(data)) {
            return DataType::DOUBLE; 
        } else {
            return DataType::STRING; 
        }
    }

    template<typename T> //Template allows for the push function to only be written once, but all types are still valid. 
    void push(const T &value) {
        if (!std::holds_alternative<std::vector<T>>(data)) {
            throw std::invalid_argument("Type Mismatch"); 
        }

        std::get<std::vector<T>>(data).push_back(value); 
    }

    void push(const char* value) {
        push(std::string(value));
    }

    size_t size() const {
        if (std::holds_alternative<std::vector<int>>(data)) {
            return std::get<std::vector<int>>(data).size(); 
        } else if (std::holds_alternative<std::vector<double>>(data)) {
            return std::get<std::vector<double>>(data).size(); 
        } else {
            return std::get<std::vector<std::string>>(data).size(); 
        }
    }

    double get_numeric(size_t idx) const {
        if (std::holds_alternative<std::vector<int>>(data)) {
            return (double) std::get<std::vector<int>>(data)[idx]; 
        } else if (std::holds_alternative<std::vector<double>>(data)) {
            return std::get<std::vector<double>>(data)[idx]; 
        } else {
            throw std::invalid_argument("String columns cannot be converted to numeric."); 
        }
    }

    std::string get_string(size_t idx) const {
        if (std::holds_alternative<std::vector<std::string>>(data)) {
            return std::get<std::vector<std::string>>(data)[idx]; 
        } else {
            throw std::invalid_argument("Numeric columns cannot be converted to string."); 
        }
    }

    int get_int(const size_t idx) const {
        if (std::holds_alternative<std::vector<int>>(data)) {
            return std::get<std::vector<int>>(data)[idx]; 
        } else {
            throw std::invalid_argument("Integer value cannot be extracted from non-integer type column."); 
        }
    }
};


class DataFrame {
private:
    std::unordered_map<std::string, Column> data; //The dataframe
public: 
    void add_column(const std::string& name, DataType type) { //Adding a column to a dataframe (checks if column already exists or not to prevent overwriting). 
        if (data.find(name) != data.end()) {
            throw std::invalid_argument("Column already Exists"); 
        }
        data[name] = Column(type); 
    }

    Column& operator[](const std::string &name) { //Modifies referencing so that columns that do not exist are not referenced. 
        auto itr = data.find(name); 
        if (itr == data.end()) {
            throw std::invalid_argument("Column Does not exist"); 
        }

        return itr -> second;  
    }

    bool is_valid() const { //Checks if the dataframe is valid currently. This allows for the push method to exist (to increase flexibility) while also keeping the invariant that all columns must be of the same size in check. 
        if (data.empty()) {
            return true; 
        }

        size_t check = data.begin() -> second.size(); 

        for (auto itr = data.begin(); itr != data.end(); itr++) {
            if (itr -> second.size() != check) {
                return false; 
            }
        }

        return true; 
    }

    size_t columns() const { //Return the number of initialized columns. 
        return data.size(); 
    }

    size_t rows() const { //If the dataframe is valid, then return the number of rows, otherwise throw an error. 
        if (!is_valid()) {
            throw std::invalid_argument("Not a valid DataFrame."); 
        }

        if (data.empty()) {
            return 0; 
        }

        return data.begin() -> second.size(); 
    }

    void add_row(const std::unordered_map<std::string, std::variant<int, double, std::string>>& row) {
        if (row.size() != data.size()) {
            throw std::invalid_argument("Row size doesn't match dataframe columns."); 
        }

        for (auto itr = row.begin(); itr != row.end(); itr++) {
            auto col_itr = data.find(itr->first);

            if (col_itr == data.end()) {
                throw std::invalid_argument("Invalid row input.");
            }

            Column& col = col_itr->second;

            if (col.type() == DataType::INT &&
                !std::holds_alternative<int>(itr->second)) {
                throw std::invalid_argument("Type mismatch.");
            }

            if (col.type() == DataType::DOUBLE &&
                !std::holds_alternative<double>(itr->second)) {
                throw std::invalid_argument("Type mismatch.");
            }

            if (col.type() == DataType::STRING &&
                !std::holds_alternative<std::string>(itr->second)) {
                throw std::invalid_argument("Type mismatch.");
            }
        }

        for (auto itr = row.begin(); itr != row.end(); itr++) {
            Column& col = data[itr -> first]; 
            if (col.type() == DataType::INT) {
                col.push(std::get<int>(itr -> second)); 
            } else if (col.type() == DataType::DOUBLE) {
                col.push(std::get<double>(itr -> second)); 
            } else if (col.type() == DataType::STRING) {
                col.push(std::get<std::string>(itr -> second)); 
            } else {
                throw std::invalid_argument("Datatype of input row does not match the column type."); 
            }
        }
    }

    Matrix to_matrix(const std::vector<std::string>& columns) const {
        if (columns.empty()) {
            throw std::invalid_argument("Empty columns list cannot be converted to matrix."); 
        }
        if (!is_valid()) {
            throw std::invalid_argument("Invalid DataFrame cannot be converted into matrix."); 
        }
        size_t cols = columns.size(), rows = data.at(columns[0]). size(); 
        Matrix result(rows, cols); 
        for (size_t i = 0; i < cols; i++) {
            const Column& currentcol = data.at(columns[i]); 
            for (size_t j = 0; j < rows; j++) {
                result(j, i) = currentcol.get_numeric(j); 
            }
        } 

        return result;
    }

    void one_hot_encode(const std::string& column) {
        std::set<std::string> unique; 
        if (data.find(column) == data.end()) {
            throw std::invalid_argument("Column does not exist."); 
        }
        Column& col = data[column]; 
        for (size_t i = 0; i < col.size(); i++) {
            unique.insert(col.get_string(i)); 
        }

        for (auto itr = unique.begin(); itr != unique.end(); itr++) {
            std::string name = column + "_" + (*itr); 
            data[name] = Column(DataType::INT); 
        }

        for (auto itr = unique.begin(); itr != unique.end(); itr++) {
            std::string name = column + "_" + (*itr);
            Column& currcol = data[name]; 
            for (size_t i = 0; i < col.size(); i++) {
                std::string value = col.get_string(i); 
                if ((*itr) == value) {
                    currcol.push(1); 
                } else {
                    currcol.push(0); 
                }
            } 
        }

        data.erase(column); 
    }

    std::vector<std::string> headers() const {
        std::vector<std::string> result; 
        for (auto itr = data.begin(); itr != data.end(); itr++) {
            result.push_back(itr -> first);
        } 

        return result; 
    }

    void shuffle(const unsigned seed = 42) {
        if (data.empty()) {
            throw std::invalid_argument("Empty dataset cannot be shuffled."); 
        }
        size_t rows = (data.begin() -> second).size();
        std::vector<size_t> indices(rows); 
        for (size_t i = 0; i < rows; i++) {
            indices[i] = i; 
        }

        std::mt19937 rng(seed); 
        std::shuffle(indices.begin(), indices.end(), rng); 

        for (auto itr = data.begin(); itr != data.end(); itr++) {      
            Column& current = itr -> second; 
            Column shuffled(current.type()); 
            DataType coltype = current.type(); 
            for (size_t i = 0; i < rows; i++) {
                if (coltype == DataType::INT) {
                    shuffled.push(current.get_int(indices[i])); 
                } else if (coltype == DataType::DOUBLE) {
                    shuffled.push(current.get_numeric(indices[i]));
                } else {
                    shuffled.push(current.get_string(indices[i])); 
                }
            }

            current = std::move(shuffled); 
        }
    }
};


class CSVReader {
public:
    static bool is_int(const std::string& s) {
        if (s.empty()) {
            return false; 
        }
        size_t start = 0; 
        if (s[0] == '-') {
            if (s.size() == 1) {
                return false; 
            }
            start = 1; 
        }
        for (size_t i = start; i < s.size(); i++) {
            if (!std::isdigit(s[i])) {
                return false;
            }
        }

        return true; 
    }

    static bool is_double(const std::string& s) {
        if (s.empty()) {
            return false; 
        }
        size_t start = 0; 
        if (s[0] == '-') {
            if (s.size() == 1) {
                return false; 
            }
            start = 1; 
        }
        size_t cnt = 0; 
        size_t digit = 0; 
        for (size_t i = start; i < s.size(); i++) {
            if (s[i] == '.') {
                cnt++;
            } else {
                if (!std::isdigit(s[i])) {
                    return false; 
                } else {
                    digit++; 
                }
            }
        }

        if (cnt > 1 || digit == 0) {
            return false;
        } else {
            return true; 
        }
    }

    static DataFrame read(const std::string& filename) {
        std::ifstream file(filename); 
        if (!file.is_open()) {
            throw std::invalid_argument("Could not open file."); 
        }
        DataFrame df;
        
        std::vector<std::vector<std::string>> raw_data; 
        std::string line; 
        while (std::getline(file, line)) {
            std::string cell;
            std::vector<std::string> row; 
            for (char ch: line) {
                if (ch == ',') {
                    row.push_back(cell); 
                    cell.clear(); 
                } else {
                    cell += ch; 
                }
            } 
            row.push_back(cell); 
            raw_data.push_back(row); 
        }

        //Verification
        if (raw_data.empty()) {
            throw std::invalid_argument("CSV file is empty.");
        }
        size_t check = raw_data[0].size(); 
        for (const std::vector<std::string>& row : raw_data) {
            if (row.size() != check) {
                throw std::invalid_argument("Invalid CSV: Inconsistent column count."); 
            }
        }

        //Making the dataframe
        std::vector<std::string> headers = raw_data[0]; 
        for (const auto& h: headers) {
            if (h.empty()) {
                throw std::invalid_argument("Empty header recieved.");
            }
        }
        for (size_t i = 0; i < headers.size(); i++) {
            DataType dtype = DataType::INT; 
            for (size_t j = 1; j < raw_data.size(); j++) { //Check (j, i)
                if (is_int(raw_data[j][i])) {
                    continue;
                } else if (is_double(raw_data[j][i])) {
                    dtype = DataType::DOUBLE;
                } else {
                    dtype = DataType::STRING; 
                    break; 
                }
            }

            df.add_column(headers[i], dtype); 
        }

        for (size_t i = 0; i < headers.size(); i++) {
            Column &col = df[headers[i]]; 
            DataType dtype = col.type(); 
            for (size_t j = 1; j < raw_data.size(); j++) {
                if (dtype == DataType::INT) {
                    col.push(std::stoi(raw_data[j][i])); 
                } else if (dtype == DataType::DOUBLE) {
                    col.push(std::stod(raw_data[j][i])); 
                } else {
                    col.push(raw_data[j][i]); 
                }
            }
        }

        if (!df.is_valid()) {
            throw std::invalid_argument("DataFrame creation failed.");
        }

        return df; 
    }
};