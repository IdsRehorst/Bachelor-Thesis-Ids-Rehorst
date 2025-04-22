/*---------------------------------------------------------------------------
 *  main.cpp  –  end‑to‑end RACE demo
 *---------------------------------------------------------------------------
 *  Build: already handled by your CMakeLists.txt
 *  Run  : ./tri_solve  matrix.mtx  [lower|upper]
 *--------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>   // std::min
#include <omp.h>
#include <RACE/interface.h>     // installed by RACE
#include <fstream>

#include "mmio.h"

/*--- Minimal CSR container using STL -----------------------------------*/
struct CSR {
    int n = 0;                       // rows == cols
    std::vector<int>    rowPtr;      // size n+1
    std::vector<int>    col;         // size nnz
    std::vector<double> val;         // size nnz
};

static bool readMM(const std::string& file, CSR& A)
{
    int nrow, ncol, nnz;
    int *I = nullptr, *J = nullptr;
    double *V = nullptr;

    if (mm_read_unsymmetric_sparse(file.c_str(),
                                   &nrow, &ncol, &nnz,
                                   &V, &I, &J) != 0) {
        std::cerr << "mm_read_unsymmetric_sparse failed for " << file << '\n';
        return false;
    }
    if (nrow != ncol) { std::cerr << "Matrix not square\n"; return false; }

    A.n = nrow;
    A.rowPtr.assign(A.n + 1, 0);

    /* --- count ------------------------------------------------------- */
    for (int k = 0; k < nnz; ++k)
        ++A.rowPtr[I[k] + 1];            // <-- +1 (critical)

    /* --- prefix sum -------------------------------------------------- */
    for (int i = 0; i < A.n; ++i)
        A.rowPtr[i + 1] += A.rowPtr[i];

    /* --- scatter ----------------------------------------------------- */
    A.col.resize(nnz);
    A.val.resize(nnz);
    std::vector<int> cursor = A.rowPtr;

    for (int k = 0; k < nnz; ++k) {
        int r = I[k];                    // already 0‑based
        int p = cursor[r]++;
        A.col[p] = J[k];
        A.val[p] = V ? V[k] : 1.0;
    }

    free(I); free(J); free(V);
    return true;
}


/*--- Keep only lower or upper triangle ---------------------------------*/
static void extractTriangle(CSR& A, bool lower)
{
    std::vector<int> newRow(A.n + 1, 0);
    for (int r = 0; r < A.n; ++r)
        for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p)
            if (lower ? (A.col[p] <= r) : (A.col[p] >= r))
                ++newRow[r+1];
    for (int r = 0; r < A.n; ++r) newRow[r+1] += newRow[r];

    std::vector<int>    newCol(newRow.back());
    std::vector<double> newVal(newRow.back());
    std::vector<int> cursor = newRow;
    for (int r = 0; r < A.n; ++r)
        for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p)
            if (lower ? (A.col[p] <= r) : (A.col[p] >= r)) {
                int q = cursor[r]++;
                newCol[q] = A.col[p];
                newVal[q] = A.val[p];
            }
    A.rowPtr.swap(newRow);
    A.col.swap(newCol);
    A.val.swap(newVal);
}

static void dumpPattern(const CSR& A,
                        const int* rowPerm,          // NULL → identity
                        const std::string& fname)
{
    std::ofstream out(fname);
    for (int r = 0; r < A.n; ++r) {
        int pr = rowPerm ? rowPerm[r] : r;           // permute row
        for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p) {
            int c = A.col[p];                        // keep original column
            out << pr << ' ' << c << '\n';
        }
    }
}

/*---------------------------------------------------------------------------
 *  permuteCSR – build B = P · A · Pᵀ so triangularity is preserved
 *---------------------------------------------------------------------------*/
static void permuteCSR(int n,
                       const CSR& A,
                       const int*   perm,          // newRow → oldRow
                       const int*   invPerm,       // oldRow → newRow
                       CSR& B)                     // output
{
    B.n = n;
    B.rowPtr.assign(n + 1, 0);

    /* count nnz per new row */
    for (int newR = 0; newR < n; ++newR)
        B.rowPtr[newR + 1] = A.rowPtr[perm[newR] + 1] - A.rowPtr[perm[newR]];

    /* prefix sum */
    for (int i = 0; i < n; ++i) B.rowPtr[i + 1] += B.rowPtr[i];

    int nnz = B.rowPtr.back();
    B.col.resize(nnz);
    B.val.resize(nnz);

    std::vector<int> cursor = B.rowPtr;
    for (int newR = 0; newR < n; ++newR) {
        int oldR = perm[newR];
        for (int p = A.rowPtr[oldR]; p < A.rowPtr[oldR + 1]; ++p) {
            int q       = cursor[newR]++;
            B.col[q]    = invPerm[A.col[p]];   // relabel column
            B.val[q]    = A.val[p];
        }
    }
}

/*----------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix.mtx> [lower|upper]\n";
        return 1;
    }
    const bool lower = (argc == 2) || (std::string(argv[2]) == "lower");
 
    CSR A;
    if (!readMM(argv[1], A)) return 1;
    std::cout << "Loaded " << argv[1] << "  (n=" << A.n
              << ", nnz=" << A.col.size() << ")\n";

    extractTriangle(A, lower);
    std::cout << "Kept " << (lower ? "lower" : "upper")
              << " triangle; nnz=" << A.col.size() << '\n';

    dumpPattern(A, /*perm*/ nullptr, "pattern_triangular.txt");
    /* --- call RACE ---------------------------------------------------- */
    RACE::Interface race(A.n, 1, RACE::ONE,
                         A.rowPtr.data(), A.col.data());

    if (race.RACEColor() != RACE_SUCCESS) {
        std::cerr << "RACEColor failed\n";  return 2;
    }

    int* perm = nullptr;
    int* inv  = nullptr;
    int  len  = 0;

    race.getPerm(&perm,     &len);
    race.getInvPerm(&inv,   &len);
 
    /* --- build permuted matrix B --------------------------------------- */
    CSR B;
    permuteCSR(A.n, A, perm, inv, B);

    /* dump in colour‑block row order, columns untouched (still triangular) */
    dumpPattern(B, /*rowPerm*/ nullptr, "pattern_after_race.txt");
    return 0;
}
