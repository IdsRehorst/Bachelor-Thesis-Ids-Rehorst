/*---------------------------------------------------------------------------
 *  tri_solve – RACE colour-blocking + triangular solve demo
 *
 *  Build : handled by CMakeLists.txt              (needs MKL + RACE + OpenMP)
 *  Run   : ./tri_solve  matrix.mtx  [lower|upper]
 *--------------------------------------------------------------------------*/

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include <omp.h>
#include <RACE/interface.h>

#include <mkl.h>
#include <mkl_spblas.h>
#include "mmio.h"
#include <chrono>
/*--------------------------------------------------------------------------*/
/* Helper Functions                                                         */
/*--------------------------------------------------------------------------*/

/** small CSR container, move to matrix class later**/
struct CSR {
    int n = 0;                       // square matrix
    std::vector<int>    rowPtr;      // n+1
    std::vector<int>    col;         // nnz
    std::vector<double> val;         // nnz
};

template<class Fn>
double time_ms(Fn &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/** We use this function to read matrixmarket files and convert them to CSR format that I will use **/
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

/* Print the levels we get after the racereordering */
void printLevels(int ncolors,
                 const int* colorPtr,
                 const int* perm)
{
    for (int c = 0; c < ncolors; ++c) {
        int first = colorPtr[c];
        int last  = colorPtr[c+1] - 1;
        std::cout << "Level " << c
                  << "  rows " << first << " … " << last
                  << "  (original indices "
                  << perm[first] << " … " << perm[last] << ")\n";
    }
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

static void solveLowerTriBlockSlice(const CSR &B,
                                    int r0, int r1,
                                    const std::vector<double> &rhs,
                                    double *x)                // slice base
{
    for (int row = r0; row < r1; ++row) {         // strict <
        double diag = 1.0;
        double s    = rhs[row - r0];

        for (int p = B.rowPtr[row]; p < B.rowPtr[row + 1]; ++p) {
            int c = B.col[p];
            if (c < row)              s    -= B.val[p] * x[c - r0];
            else if (c == row && std::fabs(B.val[p]) > 1e-30)
                                       diag  = B.val[p];
        }
        x[row - r0] = s / diag;                    // write *inside* slice
    }
}

/*==========================================================================*
 *  blockBiDiagSolve – redundant two‑phase algorithm with OpenMP tasks      *
 *                                                                          *
 *  Works for strictly lower‑triangular block–bidiagonal matrix B.          *
 *  Upper‑triangular case falls back to the sequential loop (unchanged).    *
 *==========================================================================*/
static void blockBiDiagSolve(const CSR &B, bool lower,
                             const std::vector<int> &ptr,
                             const std::vector<double> &b,
                             std::vector<double> &x)
{
    /* ---------- quick upper‑triangular fallback (sequential) ------------- */
    if (!lower) {
        int n = B.n;  x.assign(n, 0.0);
        for (int r = 0; r < n; ++r) {
            double rhs = b[r], diag = 1.0;
            for (int p = B.rowPtr[r]; p < B.rowPtr[r + 1]; ++p) {
                int c = B.col[p];
                if (c < r) rhs -= B.val[p] * x[c];
                else if (c == r && std::fabs(B.val[p]) > 1e-30) diag = B.val[p];
            }
            x[r] = rhs / diag;
        }
        return;
    }

    /* ---------- lower‑triangular block solve with redundant phase -------- */
    const int nBlocks = static_cast<int>(ptr.size()) - 1;
    const int n       = B.n;
    x.assign(n, 0.0);

#pragma omp parallel default(none) shared(B,ptr,b,x,nBlocks)
{
#pragma omp single
    {
        std::vector<int> token(nBlocks);        // 1 dummy int per block
int *tok = token.data();               // ← raw pointer (array base)

/* -------------------- phase 1 : provisional -------------------------- */
for (int blk = 0; blk < nBlocks; ++blk) {
    int r0 = ptr[blk], r1 = ptr[blk+1];
    #pragma omp task depend(out: tok[blk]) firstprivate(r0,r1,blk,tok)
    {
        std::vector<double> rhs(r1 - r0);
        std::copy(b.begin()+r0, b.begin()+r1, rhs.begin());
        solveLowerTriBlockSlice(B, r0, r1, rhs, x.data()+r0);
    }
}

/* -------------------- phase 2 : correctors --------------------------- */
for (int blk = 1; blk < nBlocks; ++blk) {
    int r0 = ptr[blk], r1 = ptr[blk+1];
    #pragma omp task depend(in: tok[blk-1]) depend(out: tok[blk]) \
                     firstprivate(r0,r1,blk,tok)
    {
        std::vector<double> rhs(r1 - r0);
        std::copy(b.begin()+r0, b.begin()+r1, rhs.begin());

        for (int row = r0; row < r1; ++row)
            for (int p = B.rowPtr[row]; p < B.rowPtr[row+1]; ++p)
                if (int c=B.col[p]; c < r0)
                    rhs[row - r0] -= B.val[p] * x[c];

        solveLowerTriBlockSlice(B, r0, r1, rhs, x.data()+r0);
    }
}

#pragma omp taskwait
    } // single
} // parallel
}

/* distance-1 level set (rows already contiguous) */
static void computeLevels(const CSR& B,
                          std::vector<int>& lev,std::vector<int>& ptr)
{
    int n=B.n; lev.assign(n,0); int Lmax=0;
    for(int r=0;r<n;++r){
        int lv=0;
        for(int p=B.rowPtr[r];p<B.rowPtr[r+1];++p)
            if(B.col[p]<r) lv=std::max(lv,lev[B.col[p]]+1);
        lev[r]=lv; Lmax=std::max(Lmax,lv);
    }
    ptr.assign(Lmax+2,0);
    for(int r=0;r<n;++r) ++ptr[lev[r]+1];
    for(int i=0;i<Lmax+1;++i) ptr[i+1]+=ptr[i];
}


static void mklTriSolve(const CSR &B, bool lower,
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

static void dumpPattern(const CSR& A, const std::string& fname)
{
    std::ofstream out(fname);
    if (!out) {
        std::perror(fname.c_str());
        return;
    }

    for (int r = 0; r < A.n; ++r)
        for (int p = A.rowPtr[r]; p < A.rowPtr[r + 1]; ++p)
            out << r << ' ' << A.col[p] << '\n';
}

int main(int argc, char* argv[])
{
    // First we check if user has inputted a matrix
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix.mtx> [lower|upper]\n";
        return 1;
    }
    const bool lower = (argc == 2) || (std::string(argv[2]) == "lower");
    
    /* --- Load the requiered matrix ------------------------------------ */
    CSR A;
    if (!readMM(argv[1], A)) return 1;
    std::cout << "Loaded " << argv[1] << "  (n=" << A.n
              << ", nnz=" << A.col.size() << ")\n";

    dumpPattern(A, "pattern");

    // Transform matrix to triangulair matrix
    //extractTriangle(A, lower);
    //dumpPattern(A, "pattern_triangular");

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

    dumpPattern(B, "pattern_after_reordering");
    /*--- 4. build level set (= block boundaries) -----------------------------*/
    std::vector<int> level, stagePtr;
    computeLevels(B, level, stagePtr);
    std::cout << "Blocks (stages) : " << stagePtr.size()-1 << '\n';

    // Transform to a triangular matrix
    extractTriangle(B, lower);
    dumpPattern(B, "pattern_after_reordering_triangular");

    /*--- 5. prepare RHS (vector of ones) -------------------------------------*/
    std::vector<double> bOrig(A.n, 1.0), bPerm(B.n);
    for (int i = 0; i < B.n; ++i) bPerm[i] = bOrig[perm[i]];

    /*--- 6. our block-bidiagonal solve ---------------------------------------*/
    std::vector<double> xPerm, xOur(A.n);
    
    double tOur = time_ms([&]{
         blockBiDiagSolve(B, lower, stagePtr, bPerm, xPerm);
    });
    for (int oldR = 0; oldR < A.n; ++oldR) xOur[oldR] = xPerm[inv[oldR]];

    /*--- 7. MKL reference solve ----------------------------------------------*/
    std::vector<double> xPermRef, xRef(A.n);

    double tMKL = time_ms([&]{
         mklTriSolve(B, lower, bPerm, xPermRef);
    });
    for (int i = 0; i < A.n; ++i) xRef[perm[i]] = xPermRef[i];

    double maxAbs = 0.0;
    for (int i = 0; i < A.n; ++i) maxAbs = std::max(maxAbs, std::abs(xOur[i] - xRef[i]));
    std::cout << "max |x_block – x_ref| = " << maxAbs << std::endl;
    std::cout << "first 10 x (ours vs MKL):";
    for (int i = 0; i < std::min(10, A.n); ++i)
        std::cout << " (" << xOur[i] << "," << xRef[i] << ")";
    std::cout << std::endl;

    std::cout << "timings  (OpenMP threads = "
          << omp_get_max_threads()      // prints the actual number used
          << ", MKL threads = " << mkl_get_max_threads() << ")\n"
          << "  our solver : " << tOur << " ms\n"
          << "  MKL        : " << tMKL << " ms\n"
          << "  speed‑up   : " << tMKL / tOur << " ×\n";

    delete[] perm; delete[] inv;
    return 0;
}



