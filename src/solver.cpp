#include "../include/solver.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <mkl_spblas.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <likwid.h>

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

double solver::mklTriSolve(const sparsemat &B, bool lower,
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

    /// Hoe kan ik hier nog ergens thread init in fietsen?
    // Init Likwid marker for measuring perfomance
    LIKWID_MARKER_START("MKL");

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

    LIKWID_MARKER_STOP("MKL");

    std::cout <<"MKL (without setup)" << tMKL << "ms"<< std::endl;
 return tMkl;
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

void solver::blockBiDiagSolveTasksAffinity(const sparsemat&           B,
                                   const std::vector<int>&    stagePtr,
                                   const std::vector<double>& b,
                                   std::vector<double>&       x)
{	
    const int k = int(stagePtr.size()) - 1;
    const int N = B.n;
    x.assign(N, 0.0);

    /* raw pointers so that OpenMP array–section syntax works */
    double       *xp = x.data();
    const double *bp = b.data();

#pragma omp parallel default(none) shared(B,stagePtr,bp,xp,k)
{
	// Needed because otherwise we only get insight in one thread
        LIKWID_MARKER_THREADINIT;
	// Init Likwid marker for measuring perfomance
        LIKWID_MARKER_START("sptrsv");
#pragma omp single
{
    /* ---------------- Phase 1 : provisional solves ---------------- */
    for (int i = 0; i < k; ++i) {
        const int r0 = stagePtr[i];
        const int r1 = stagePtr[i+1];
        const int m  = r1 - r0;                 /* block size            */

        /* length in the array-section must be  r1-r0,  not the last index */
#pragma omp task                                                \
        depend(out: xp[r0:m])                                   \
        affinity( xp[r0:m] )                                      \
        firstprivate(r0,r1,m)
        {
            std::vector<double> rhs(m), xi(m);

            /* RHS = b_i */
            for (int j = 0; j < m; ++j)
                rhs[j] = bp[r0 + j];

            /* solve L_i · x̂ = rhs */
            for (int ii = 0; ii < m; ++ii) {
                int    row  = r0 + ii;
                double sum  = rhs[ii];
                double diag = 1.0;

                for (int p = B.rowPtr[row]; p < B.rowPtr[row+1]; ++p) {
                    int c = B.col[p];
                    if      (c <  r0) continue;            /* belongs to B_i   */
                    else if (c <  row) sum  -= B.val[p] * xi[c - r0];
                    else if (c == row) diag  = B.val[p];
                }
                assert(std::abs(diag) > 1e-30);
                xi[ii] = sum / diag;
            }

            /* write provisional result */
            for (int j = 0; j < m; ++j) xp[r0 + j] = xi[j];
        }
    }

    /* ---------------- Phase 2 : correction solves ----------------- */
    for (int i = 1; i < k; ++i) {
        const int r0 = stagePtr[i];
        const int r1 = stagePtr[i+1];
        const int m  = r1 - r0;

#pragma omp task  depend(in:    xp[stagePtr[i-1]:stagePtr[i]-stagePtr[i-1]]) \
                  depend(inout: xp[r0:m])                        \
                  affinity( xp[r0 : m])                             \
                  firstprivate(r0,r1,m)
        {
            std::vector<double> rhs(m), xi(m);

            /* RHS = b_i – B_i · x_{i-1} */
            for (int ii = 0; ii < m; ++ii) {
                int    row = r0 + ii;
                double sum = bp[row];

                for (int p = B.rowPtr[row]; p < B.rowPtr[row+1]; ++p) {
                    int c = B.col[p];
                    if (c < r0) sum -= B.val[p] * xp[c];
                }
                rhs[ii] = sum;
            }

            /* solve L_i · x = rhs */
            for (int ii = 0; ii < m; ++ii) {
                int    row  = r0 + ii;
                double sum  = rhs[ii];
                double diag = 1.0;

                for (int p = B.rowPtr[row]; p < B.rowPtr[row+1]; ++p) {
                    int c = B.col[p];
                    if      (c <  r0) continue;            /* already in RHS   */
                    else if (c <  row) sum  -= B.val[p] * xi[c - r0];
                    else if (c == row) diag  = B.val[p];
                }
                assert(std::abs(diag) > 1e-30);
                xi[ii] = sum / diag;
            }

            /* write corrected result */
            for (int j = 0; j < m; ++j) xp[r0 + j] = xi[j];
        }
    }
    /* implicit taskwait here */
} /* single */
LIKWID_MARKER_STOP("sptrsv");
} /* parallel */
}

#include <Kokkos_Core.hpp>
#include <KokkosSparse_sptrsv.hpp>
#include <KokkosKernels_Handle.hpp>
#include <chrono>          // <-- for timing

double solver::kokkosSpTRSV(const sparsemat&           B,
                            const std::vector<double>& b,
                            std::vector<double>&       x)
{
  using exec_space = Kokkos::DefaultExecutionSpace;          // OpenMP here
  using mem_space  = typename exec_space::memory_space;

  using size_type  = int;      // match your CSR index types
  using lno_type   = int;
  using scalar     = double;

  const size_type n   = B.n;
  const size_type nnz = static_cast<size_type>(B.val.size());

  /* ------------ host views (build once per call) ---------------- */
  using RowH = Kokkos::View<size_type*, Kokkos::HostSpace>;
  using ColH = Kokkos::View<lno_type*,  Kokkos::HostSpace>;
  using ValH = Kokkos::View<scalar*,    Kokkos::HostSpace>;

  RowH row_h("row_h", n + 1);
  ColH col_h("col_h", nnz);
  ValH val_h("val_h", nnz);

  for (size_type i = 0; i <= n;  ++i) row_h(i) = B.rowPtr[i];
  for (size_type p = 0; p <  nnz; ++p) {
    col_h(p) = B.col[p];
    val_h(p) = B.val[p];
  }

  /* ------------ device mirrors ---------------------------------- */
  using RowD = Kokkos::View<size_type*, mem_space>;
  using ColD = Kokkos::View<lno_type*,  mem_space>;
  using ValD = Kokkos::View<scalar*,    mem_space>;
  using VecD = Kokkos::View<scalar*,    mem_space>;

  RowD row_d("row_d", n + 1);   ColD col_d("col_d", nnz);
  ValD val_d("val_d", nnz);     VecD b_d  ("b_d",   n);
  VecD x_d  ("x_d",   n);

  Kokkos::deep_copy(row_d, row_h);
  Kokkos::deep_copy(col_d, col_h);
  Kokkos::deep_copy(val_d, val_h);

  { // copy RHS once
    auto b_h = Kokkos::create_mirror_view(b_d);
    for (size_type i = 0; i < n; ++i) b_h(i) = b[i];
    Kokkos::deep_copy(b_d, b_h);
  }

  /* ------------ kernel-handle + symbolic ------------------------ */
  auto t0 = std::chrono::high_resolution_clock::now();
  using KH = KokkosKernels::Experimental::KokkosKernelsHandle<
               size_type, lno_type, scalar,
               exec_space, mem_space, mem_space>;

  KH kh;
  kh.create_sptrsv_handle(
        KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1,
        n, /*is_lower =*/ true);

  KokkosSparse::Experimental::sptrsv_symbolic(&kh, row_d, col_d, val_d);
  exec_space().fence();   // done with symbolic part

  /* ------------ NUMERIC solve (timed) --------------------------- */

  KokkosSparse::Experimental::sptrsv_solve(&kh,
          row_d, col_d, val_d,
          b_d,  x_d);

  exec_space().fence();
  auto t1 = std::chrono::high_resolution_clock::now();
  const double t_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  /* ------------ copy result back -------------------------------- */
  auto x_h = Kokkos::create_mirror_view(x_d);
  Kokkos::deep_copy(x_h, x_d);
  x.assign(x_h.data(), x_h.data() + n);

  kh.destroy_sptrsv_handle();
  return t_ms;                    // <----  numeric time only
}
