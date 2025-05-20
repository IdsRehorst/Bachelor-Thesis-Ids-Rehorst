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

#include "sparsemat.h"
#include "coloring.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include "mmio.h"
#include <chrono>

#include "solver.h"

template<class Fn>
double time_ms(Fn &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
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
    
    /* --- Load the required matrix ------------------------------------ */
    sparsemat A;
    if (!A.load_matrix_market(argv[1])) return 1;
    std::cout << "Loaded " << argv[1] << "  (n=" << A.n
              << ", nnz=" << A.col.size() << ")\n";

    A.dump_pattern( "pattern");
    A.extract_triangle(true);
    // compute colouring + permuted matrix + levels
    coloring col;
    col.compute(A);

    sparsemat B = col.permuted;
    B.extract_triangle(lower);

    // solve the whole triangular matrix
    std::vector<double> b(B.n, 1.0);

    // 4) Solve with MKL
    std::vector<double> xMkl;
    solver::mklTriSolve(B, /*lower=*/true, b, xMkl);

    // 5) Solve with block‐bidiagonal
    std::vector<double> xBlk;
    solver::blockBiDiagSolve(B, col.stagePtr, b, xBlk);

    // 3) plain serial SpTRSV
    std::vector<double> xSerial;
    solver::serialSpTRSV(B, true, b, xSerial);

    // 6) Compute and print the error
    double err = solver::maxAbsError(xMkl, xBlk);
    double errSerial = solver::maxAbsError(xMkl, xSerial);
    std::cout << "Max |x_MKL – x_serial|   = " << errSerial << "\n\n";
    std::cout << "Max |x_MKL – x_block| = " << err << "\n";

    solver::printErrorSummary(xMkl, xBlk, 10);

    return 0;
}



