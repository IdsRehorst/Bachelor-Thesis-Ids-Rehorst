#include "../include/solver.h"

#include <algorithm>
#include <cassert>
#include <mkl_spblas.h>
#include <chrono>
#include <cmath>
#include <iostream>

#include "coloring.h"

// Quick and easy method to time stuff
template<class Fn>
double time_ms(Fn &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void solver::mklTriSolve(const sparsemat &B, bool lower,
                         const std::vector<double> &b,
                         std::vector<double> &x)
{
    int n = B.n;

    // --- check if every row has an explicit diagonal --------------------
    bool explicitDiag = true;
    for (int r = 0; r < n && explicitDiag; ++r) {
        bool found = false;
        for (int p = B.rowPtr[r]; p < B.rowPtr[r + 1]; ++p)
            if (B.col[p] == r) { found = true; break; }
        explicitDiag = found;
    }


    sparse_matrix_t A = nullptr;
    matrix_descr desc{};
    desc.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    desc.mode = lower ? SPARSE_FILL_MODE_LOWER : SPARSE_FILL_MODE_UPPER;
    desc.diag = explicitDiag ? SPARSE_DIAG_NON_UNIT : SPARSE_DIAG_UNIT;

    std::vector<MKL_INT> ia(n + 1);  std::vector<MKL_INT> ja(B.col.size());
    for (int i = 0; i <= n; ++i) ia[i] = B.rowPtr[i];
    for (size_t k = 0; k < B.col.size(); ++k) ja[k] = B.col[k];

    double tMKL = time_ms([&]{

        mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO,
                            n, n, ia.data(), ia.data() + 1,
                            ja.data(), const_cast<double*>(B.val.data()));
        mkl_sparse_optimize(A);

        x.assign(n, 0.0);
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,
                      1.0, A, desc,
                      b.data(), x.data());
        mkl_sparse_destroy(A);
    });
    std::cout <<"MKL (without setup)" << tMKL << "ms"<< std::endl;

}

//------------------------------------------------------------------------------
// blockBiDiagSolve : sequential block forward‐substitution on CSR B
//      - B must already be the RACE‐permuted & triangular matrix
//      - stagePtr has size k+1, giving the row range [stagePtr[i],stagePtr[i+1])
//------------------------------------------------------------------------------
void solver::blockBiDiagSolve(
    const sparsemat&        B,
    const std::vector<int>& stagePtr,
    const std::vector<double>& b,
    std::vector<double>&       x)
{
    int k = int(stagePtr.size()) - 1;
    int N = B.n;
    x.assign(N, 0.0);

    std::vector<double> rhs, xi;
    // Loop over each block i
    for (int i = 0; i < k; ++i) {
        int r0 = stagePtr[i];
        int r1 = stagePtr[i+1];
        int m  = r1 - r0;

        // 1) Build RHS = b_i  minus all previous‐row couplings
        rhs.resize(m);
        for (int j = 0; j < m; ++j)
            rhs[j] = b[r0 + j];

        // Subtract for row = r0..r1-1, any col < r0
        for (int row = r0; row < r1; ++row) {
            double sum = 0.0;
            for (int p = B.rowPtr[row]; p < B.rowPtr[row+1]; ++p) {
                int c = B.col[p];
                if (c < r0) {
                    sum += B.val[p] * x[c];
                }
            }
            rhs[row - r0] -= sum;
        }

        // 2) Forward‐solve the small diagonal block L_i · xi = rhs
        xi.assign(m, 0.0);
        for (int ii = 0; ii < m; ++ii) {
            int row = r0 + ii;
            double s    = rhs[ii];
            double diag = 1.0;

            // scan only the block‐rows [r0..r1)
            for (int p = B.rowPtr[row]; p < B.rowPtr[row+1]; ++p) {
                int c = B.col[p];
                if (c < r0) {
                    // already handled in RHS
                }
                else if (c < row) {
                    // within-block subdiagonal
                    s -= B.val[p] * xi[c - r0];
                }
                else if (c == row) {
                    // the diagonal entry
                    diag = B.val[p];
                }
            }
            // must have a nonzero diagonal
            assert(std::abs(diag) > 1e-30);
            xi[ii] = s / diag;
        }

        // 3) Scatter block solution back into x
        for (int j = 0; j < m; ++j)
            x[r0 + j] = xi[j];
    }
}

double solver::maxAbsError(const std::vector<double>& xA,
                           const std::vector<double>& xB)
{
    assert(xA.size() == xB.size());
    double maxErr = 0.0;
    for (size_t i = 0; i < xA.size(); ++i) {
        double err = std::abs(xA[i] - xB[i]);
        if (err > maxErr) maxErr = err;
    }
    return maxErr;
}

void solver::printFirst(const std::string &label,
                        const std::vector<double> &x,
                        size_t count)
{
    size_t n = std::min(count, x.size());
    std::cout << label << " (first " << n << " entries):";
    for (size_t i = 0; i < n; ++i) {
        std::cout << ' ' << x[i];
    }
    std::cout << '\n';
}

void solver::blockBiDiagSolveExtract(const sparsemat&        B,
                                     const std::vector<int>& stagePtr,
                                     const std::vector<double>& b,
                                     std::vector<double>&       x)
{
    int k = int(stagePtr.size()) - 1;
    int N = B.n;
    x.assign(N, 0.0);

    // 1) Extract Lblocks[i] and Bblocks[i]
    std::vector<sparsemat> Lblocks(k), Bblocks(k);
    coloring col;                    // reuse extract_blocks logic

    col.stagePtr = stagePtr;
    col.extract_blocks(B, Lblocks, Bblocks);

    // 2) Temporary vectors
    std::vector<double> rhs, xi;

    // 3) Loop over blocks
    for (int i = 0; i < k; ++i) {
        int r0 = stagePtr[i], r1 = stagePtr[i+1];
        int m  = r1 - r0;

        // 3a) build rhs
        rhs.assign(m, 0.0);
        for (int j = 0; j < m; ++j)
            rhs[j] = b[r0 + j];

        // subtract B_i * x_{i-1}
        if (i > 0) {
            auto &Bi = Bblocks[i];
            for (int row = 0; row < m; ++row) {
                for (int p = Bi.rowPtr[row]; p < Bi.rowPtr[row+1]; ++p) {
                    int c = Bi.col[p];      // col index in block i-1
                    rhs[row] -= Bi.val[p] * x[ stagePtr[i-1] + c ];
                }
            }
        }

        // 3b) solve Li * xi = rhs via MKL
        solver::mklTriSolve(Lblocks[i], /*lower=*/true, rhs, xi);

        // 3c) scatter back into global x
        for (int j = 0; j < m; ++j)
            x[r0 + j] = xi[j];
    }
}

void solver::printErrorSummary(
    const std::vector<double>& xA,
    const std::vector<double>& xB,
    size_t count)
{
    assert(xA.size() == xB.size());
    size_t N = xA.size();

    // collect (error, index) pairs
    std::vector<std::pair<double,size_t>> errs;
    errs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        errs.emplace_back( std::abs(xA[i] - xB[i]), i );
    }

    // find maximum
    auto itMax = std::max_element(
        errs.begin(), errs.end(),
        [](auto &a, auto &b){ return a.first < b.first; }
    );
    double maxErr = itMax->first;
    size_t maxIdx = itMax->second;
    std::cout << "→ Max error = " << maxErr
              << " at index " << maxIdx
              << " (xA=" << xA[maxIdx]
              << ", xB=" << xB[maxIdx] << ")\n\n";

    // sort descending by error
    std::sort(errs.begin(), errs.end(),
        [](auto &a, auto &b){ return a.first > b.first; }
    );

    // print top `count` errors
    size_t toShow = std::min(count, errs.size());
    std::cout << "Top " << toShow << " errors:\n";
    for (size_t k = 0; k < toShow; ++k) {
        double e = errs[k].first;
        size_t idx = errs[k].second;
        std::cout << "  [" << idx << "] err=" << e
                  << "   xA=" << xA[idx]
                  << "   xB=" << xB[idx] << "\n";
    }
    std::cout << std::endl;
}





void solver::blockBiDiagSolve(
    const std::vector<sparsemat>& Lblocks,
    const std::vector<sparsemat>& Bblocks,
    const std::vector<int>&       stagePtr,
    const std::vector<double>&    b,
    std::vector<double>&          x)
{
    int k = int(Lblocks.size());
    assert((int)Bblocks.size() == k);
    assert((int)stagePtr.size() == k+1);

    int N = stagePtr.back();
    x.assign(N, 0.0);

    // scratch space
    std::vector<double> rhs, xi;

    // loop over each block
    for (int i = 0; i < k; ++i) {
        int r0 = stagePtr[i], r1 = stagePtr[i+1];
        int m  = r1 - r0;

        // 1) build the right‐hand side
        rhs.resize(m);
        for (int j = 0; j < m; ++j)
            rhs[j] = b[r0 + j];

        if (i > 0) {
            // subtract B_i · x_{i-1}
            const sparsemat &Bi = Bblocks[i];
            int m_prev = stagePtr[i] - stagePtr[i-1];
            for (int row = 0; row < m; ++row) {
                double sum = 0.0;
                for (int p = Bi.rowPtr[row]; p < Bi.rowPtr[row+1]; ++p) {
                    int c = Bi.col[p];         // 0 ≤ c < m_prev
                    sum += Bi.val[p] * x[stagePtr[i-1] + c];
                }
                rhs[row] -= sum;
            }
        }

        // 2) solve L_i · xi = rhs
        const sparsemat &Li = Lblocks[i];
        xi.assign(m, 0.0);
        for (int row = 0; row < m; ++row) {
            double s = rhs[row], diag = 1.0;
            for (int p = Li.rowPtr[row]; p < Li.rowPtr[row+1]; ++p) {
                int c = Li.col[p];
                if (c < row)        s -= Li.val[p] * xi[c];
                else if (c == row)  diag = Li.val[p];
            }
            assert(std::abs(diag) > 1e-30);
            xi[row] = s / diag;
        }

        // 3) scatter back into global x
        for (int j = 0; j < m; ++j)
            x[r0 + j] = xi[j];
    }
}

void solver::serialSpTRSV(const sparsemat& B,
                          bool lower,
                          const std::vector<double>& b,
                          std::vector<double>& x)
{
    int n = B.n;
    x.assign(n, 0.0);

    // For each row 0..n-1 (we assume B is already lower-triangular
    // in CSR order).  If upper, you can reverse the loops similarly.
    for (int r = 0; r < n; ++r) {
        double sum  = b[r];
        double diag = 1.0;

        // scan the sparse row:
        for (int p = B.rowPtr[r]; p < B.rowPtr[r+1]; ++p) {
            int c = B.col[p];
            double v = B.val[p];
            if (c < r) {
                sum -= v * x[c];
            }
            else if (c == r) {
                diag = v;
            }
        }

        // sanity
        assert(std::fabs(diag) > 1e-30 && "zero diagonal");
        x[r] = sum / diag;
    }
}