//
// Created by Ids Rehorst on 19/05/2025.
//

#ifndef SPARSEMAT_H
#define SPARSEMAT_H

#include <vector>
#include <string>

class sparsemat
{
public:
    using index_t = int;
    using value_t = double;

    // constructors
    sparsemat()  = default;
    explicit sparsemat(index_t n) : n(n), rowPtr(n + 1, 0) {}

    int n = 0;                       // square matrix
    std::vector<int>    rowPtr;      // n+1
    std::vector<int>    col;         // nnz
    std::vector<double> val;         // nnz

    // Load a (square) MatrixMarket file; returns true on success
    bool load_matrix_market(const std::string& filename);

    // Write the (row, col) pattern to a text file
    void dump_pattern(const std::string& filename) const;

    // Keep only lower (default) or upper triangle, in-place.
    void extract_triangle(bool lower = true);
};

#endif //SPARSEMAT_H
