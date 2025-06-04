//
// Created by Ids Rehorst on 19/05/2025.
//

#include "../include/sparsemat.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>          // free()
#include <numeric>
#include <unordered_map>

#include "mmio.h"           // MatrixMarket reader

bool sparsemat::load_matrix_market(const std::string &file)
{
    FILE *f = fopen(file.c_str(), "r");
    if (!f) {
        std::perror(file.c_str());
        return false;
    }

    MM_typecode code;
    if (mm_read_banner(f, &code) != 0) {
        std::cerr << "Invalid MatrixMarket banner: " << file << "\n";
        fclose(f);
        return false;
    }

    if (!mm_is_coordinate(code) ||
        !(mm_is_real(code) || mm_is_pattern(code) || mm_is_integer(code)))
    {
        std::cerr << "Only coordinate real/pattern/integer supported\n";
        fclose(f);
        return false;
    }

    bool isPattern = mm_is_pattern(code);
    bool isSymm    = mm_is_symmetric(code)
                   || mm_is_hermitian(code)
                   || mm_is_skew(code);

    int nrow, ncol, nnzIn;
    if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnzIn) != 0) {
        std::cerr << "Failed to read size line\n";
        fclose(f);
        return false;
    }
    if (nrow != ncol) {
        std::cerr << "Matrix must be square\n";
        fclose(f);
        return false;
    }

    // allocate temporary arrays for the input entries
    std::vector<int>   TI(nnzIn), TJ(nnzIn);
    std::vector<double> TV(nnzIn);

    // read all coordinate data in one call
    if (mm_read_mtx_crd_data(f,
                             nrow, ncol, nnzIn,
                             TI.data(), TJ.data(),
                             TV.data(),
                             code) != 0)
    {
        std::cerr << "Failed reading matrix data\n";
        fclose(f);
        return false;
    }
    fclose(f);

    // convert 1-based → 0-based, and duplicate symmetric entries
    std::vector<int>   I, Jv;
    std::vector<double> V;
    I.reserve(isSymm ? nnzIn*2 : nnzIn);
    Jv.reserve(isSymm ? nnzIn*2 : nnzIn);
    V .reserve(isSymm ? nnzIn*2 : nnzIn);

    for (int k = 0; k < nnzIn; ++k) {
        int i = TI[k] - 1;
        int j = TJ[k] - 1;
        double val_ = isPattern ? 1.0 : TV[k];

        I .push_back(i);
        Jv.push_back(j);
        V .push_back(val_);

        if (isSymm && i != j) {
            I .push_back(j);
            Jv.push_back(i);
            V .push_back(val_);
        }
    }

    // build CSR
    int N   = nrow;
    int nnz = (int)I.size();
    n = N;
    rowPtr.assign(N+1, 0);
    for (int k = 0; k < nnz; ++k) {
        ++rowPtr[I[k] + 1];
    }
    for (int i = 0; i < N; ++i) {
        rowPtr[i+1] += rowPtr[i];
    }

    col.resize(nnz);
    val.resize(nnz);
    std::vector<int> cursor = rowPtr;
    for (int k = 0; k < nnz; ++k) {
        int r = I[k], dest = cursor[r]++;
        col[dest] = Jv[k];
        val[dest] = V[k];
    }

    return true;
}

// extract_triangle
// We process the matrix so we are sure it will always be solved right by intelMKL (so that it knows what to expect)
void sparsemat::extract_triangle(bool lower)
{
    // 1) First count how many nonzeros per row we’ll keep,
    //    + also note if the diagonal was present.
    std::vector<index_t> newRow(n+1, 0);
    std::vector<bool>    hasDiag(n, false);

    for (index_t r = 0; r < n; ++r) {
        for (index_t p = rowPtr[r]; p < rowPtr[r+1]; ++p) {
            index_t c = col[p];
            if (lower ? (c <= r) : (c >= r)) {
                ++newRow[r+1];
                if (c == r) {
                    hasDiag[r] = true;
                }
            }
        }
        // if diagonal was missing, we’ll need one extra slot:
        if (!hasDiag[r]) {
            ++newRow[r+1];
        }
    }

    // 2) Prefix‐sum to build newRow (CSR row pointers)
    for (index_t r = 0; r < n; ++r) {
        newRow[r+1] += newRow[r];
    }

    // 3) Allocate new storage
    index_t nnzNew = newRow.back();
    std::vector<index_t> newCol(nnzNew);
    std::vector<value_t> newVal(nnzNew);
    std::vector<index_t> cursor = newRow;

    // 4) Fill in the kept entries (and implicit diagonals)
    for (index_t r = 0; r < n; ++r) {
        // (a) copy all the off-diagonal + diagonal entries we keep
        for (index_t p = rowPtr[r]; p < rowPtr[r+1]; ++p) {
            index_t c = col[p];
            if (lower ? (c <= r) : (c >= r)) {
                index_t q = cursor[r]++;
                newCol[q] = c;
                newVal[q] = val[p];
            }
        }
        // (b) if diagonal was missing, insert it with value=1.0
        if (!hasDiag[r]) {
            index_t q = cursor[r]++;
            newCol[q] = r;
            newVal[q] = 1.0;
        }
    }

    // 5) Now sort each row by column index (so MKL, etc. get ascending cols)
    for (index_t r = 0; r < n; ++r) {
        index_t p0 = newRow[r], p1 = newRow[r+1];
        if (p1 - p0 <= 1) continue;
        std::vector<std::pair<index_t, value_t>> tmp;
        tmp.reserve(p1-p0);
        for (index_t p = p0; p < p1; ++p) {
            tmp.emplace_back(newCol[p], newVal[p]);
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });
        for (index_t p = p0; p < p1; ++p) {
            newCol[p] = tmp[p-p0].first;
            newVal[p] = tmp[p-p0].second;
        }
    }

    // 6) Commit the new CSR
    rowPtr.swap(newRow);
    col   .swap(newCol);
    val   .swap(newVal);
}

// dump_pattern
void sparsemat::dump_pattern(const std::string& fname) const
{
    std::ofstream out(fname);
    if (!out)
    {
        std::perror(fname.c_str());
        return;
    }

    for (index_t r = 0; r < n; ++r)
        for (index_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p)
            out << r << ' ' << col[p] << '\n';
}

sparsemat sparsemat::multiply(const sparsemat& B) const
{
    // -------- 1. basic sanity ---------------------------------------------
    if (col.empty() || B.col.empty())
        return {};                          // empty result

    const int M = n;                        // rows  of A  ( = rows of C )
    const int K = n;                        // cols  of A / rows of B
    const int N = B.n;                      // cols  of B  ( = cols of C )

    sparsemat C;
    C.n = M;
    C.rowPtr.assign(M + 1, 0);

    std::vector<index_t>   tmpCols;         // collects columns per row (sorted later)
    std::vector<value_t>   tmpVals;

    // -------- 2. row-by-row multiplication -------------------------------
    std::unordered_map<index_t, value_t> accum;     // <col → value>
    accum.reserve(64);

    for (int i = 0; i < M; ++i)
    {
        accum.clear();

        // A’s row i
        for (int p = rowPtr[i]; p < rowPtr[i + 1]; ++p)
        {
            const int    k   = col[p];
            const double aik = val[p];

            // B’s row k  (because B is CSR too)
            for (int q = B.rowPtr[k]; q < B.rowPtr[k + 1]; ++q)
            {
                const int    j   = B.col[q];
                const double bkj = B.val[q];
                accum[j] += aik * bkj;
            }
        }

        // copy accum → C
        const int nnzRow = static_cast<int>(accum.size());
        C.rowPtr[i + 1] = C.rowPtr[i] + nnzRow;

        // append, but keep columns sorted (CSR requirement)
        if (nnzRow)
        {
            tmpCols.resize(nnzRow);
            tmpVals.resize(nnzRow);
            int idx = 0;
            for (const auto& kv : accum)        // unordered_map gives arbitrary order
            {
                tmpCols[idx] = kv.first;
                tmpVals[idx] = kv.second;
                ++idx;
            }
            std::vector<int> perm(nnzRow);
            std::iota(perm.begin(), perm.end(), 0);
            std::sort(perm.begin(), perm.end(),
                      [&](int a, int b){ return tmpCols[a] < tmpCols[b]; });

            for (int p : perm)
            {
                C.col.push_back(tmpCols[p]);
                C.val.push_back(tmpVals[p]);
            }
        }
    }
    return C;
}

void sparsemat::make_unit_diagonal()
{
    // We may need to insert up to n new non-zeros, so reserve once.
    col.reserve(col.size() + n);
    val.reserve(val.size() + n);

    // Walk rows *backwards* so that rowPtr stays valid while we insert.
    for (int r = n - 1; r >= 0; --r)
    {
        bool found = false;
        int  insertPos = rowPtr[r+1];   // default: append at end of the row

        // Scan the current row
        for (int p = rowPtr[r]; p < rowPtr[r+1]; ++p)
        {
            if (col[p] == r)            // diagonal already present
            {
                val[p] = 1.0;
                found  = true;
                break;
            }
            if (col[p] > r && insertPos == rowPtr[r+1])
                insertPos = p;          // remember first col > r ⇒ keep CSR sorted
        }

        if (found) continue;            // nothing more to do for this row

        /* --- Insert a new (r,r,1.0) at insertPos ----------------------- */

        col.insert(col.begin() + insertPos, r);
        val.insert(val.begin() + insertPos, 1.0);

        // Shift rowPtr for all subsequent rows
        for (int i = r + 1; i <= n; ++i) ++rowPtr[i];
    }
}