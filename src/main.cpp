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

static void writePattern(const CSR& A,
                         const int* perm, const std::string& fname)
{
    std::ofstream out(fname);
    for (int r = 0; r < A.n; ++r) {
        int pr = perm ? perm[r] : r;              // apply permutation
        for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p) {
            int pc = perm ? perm[A.col[p]] : A.col[p];
            out << pr << ' ' << pc << '\n';
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

    /* --- call RACE ---------------------------------------------------- */
    RACE::Interface race(A.n, 1, RACE::ONE,
                         A.rowPtr.data(), A.col.data());

    if (race.RACEColor() != RACE_SUCCESS) {
        std::cerr << "RACEColor failed\n";  return 2;
    }

    int* perm = nullptr;
    int  len  = 0;
    race.getPerm(&perm, &len);

    std::cout << "Stages: " << race.getMaxStageDepth()+1
          << "   first 10 perm:";
    for (int i = 0; i < std::min(10, len); ++i) std::cout << ' ' << perm[i];
    std::cout << '\n';
	
    writePattern(A, perm, "pattern_after_race.txt");

    return 0;
}
