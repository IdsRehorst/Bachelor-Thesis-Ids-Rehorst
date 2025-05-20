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
    std::vector<std::string> mats = {
        "../matrices/3elt/3elt.mtx",
        "../matrices/thermal2/thermal2.mtx",
        "../matrices/spinSZ12.mm", /*…*/
        "../matrices/crankseg_1/crankseg_1.mtx",
        "../matrices/F1/F1.mtx",
        "../matrices/Fault_63/Fault_63.mtx",
        "../matrices/nlpkkt200/nlpkkt200.mtx",
        "../matrices/offshore/offshore.mtx",
        "../matrices/pwtk/pwtk.mtx",
        "../matrices/Serena/Serena.mtx",
        "../matrices/ship_003/ship_003.mtx",
        "../matrices/thermal2/thermal2.mtx"
      };

    std::ofstream out("benchmark.csv");
    out << "matrix,n,nnz,t_mkl_ms,t_tasks_ms,speedup\n";

    for (auto &file : mats) {
        sparsemat A;   A.load_matrix_market(file);
        A.extract_triangle(true);
        coloring col;  col.compute(A);
        sparsemat B = col.permuted;
        B.extract_triangle(true);

        // extract blocks once
        std::vector<sparsemat> Lblocks, Bblocks;
        col.extract_blocks(B, Lblocks, Bblocks);

        // prepare RHS
        std::vector<double> bOrig(A.n,1.0), bPerm(A.n);
        for(int i=0;i<A.n;++i) bPerm[i] = bOrig[col.perm[i]];

        // warm-up
        std::vector<double> x1, x2;
        solver::mklTriSolve   (B,true,bPerm,x1);
        solver::blockBiDiagSolveTasks(B,col.stagePtr,bPerm,x2);

        // time both solvers K times
        const int K = 5;
        double t_mkl=0, t_tasks=0;
        for(int i=0;i<K;++i) {
            t_mkl   += time_ms([&]{ solver::mklTriSolve   (B,true,bPerm,x1); });
            t_tasks += time_ms([&]{ solver::blockBiDiagSolveTasks(B,col.stagePtr,bPerm,x2); });
        }
        t_mkl   /= K;
        t_tasks /= K;

        double speedup = t_mkl / t_tasks;
        out << file << ","
            << A.n  << ","
            << A.col.size() << ","
            << t_mkl   << ","
            << t_tasks << ","
            << speedup << "\n";
        std::cout << "Done " << file << "\n";
    }

    return 0;
}



