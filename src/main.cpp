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
#include <likwid.h>
#include <omp.h>
#include <RACE/interface.h>

#include "sparsemat.h"
#include "coloring.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include "mmio.h"
#include <chrono>
#include <Kokkos_Core.hpp>

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

    // Initialise Likwid marker
    LIKWID_MARKER_INIT;
    
    Kokkos::initialize();
    
    std::vector<std::string> mats = {
        "matrices/3elt/3elt.mtx",
        "matrices/thermal2/thermal2.mtx",
        "matrices/spinSZ12.mm",
        "matrices/crankseg_1/crankseg_1.mtx",
        "matrices/F1/F1.mtx",
        "matrices/Fault_639/Fault_639.mtx",
        "matrices/nlpkkt200/nlpkkt200.mtx",
        "matrices/offshore/offshore.mtx",
        "matrices/pwtk/pwtk.mtx",
        "matrices/Serena/Serena.mtx",
        "matrices/ship_003/ship_003.mtx",
        "matrices/channel-500x100x100-b050/channel-500x100x100-b050.mtx",
        "matrices/delaunay_n22/delaunay_n22.mtx",
        "matrices/delaunay_n23/delaunay_n23.mtx",
        "matrices/delaunay_n24/delaunay_n24.mtx",
        "matrices/nlpkkt120/nlpkkt120.mtx",
        "matrices/nlpkkt160/nlpkkt160.mtx",
        "matrices/G3_circuit/G3_circuit.mtx",
    	//"matrices/mawi_201512020330/mawi_201512020330.mtx",
        "matrices/Spielman_k500/Spielman_k500_A_09.mtx"
      };

    std::ofstream out(argv[1]);
    out << "matrix,n,nnz,nzr,t_mkl_ms,t_tasks_ms, t_trilinos_ms, speedup_tasks, speedup_mkl\n";

    for (auto &file : mats) {
        sparsemat A; 
      	A.load_matrix_market(file);

	// We take A^3 to make the matrices less sparse
	A.extract_triangle(true);
	A = A.multiply(A).multiply(A);
        A.extract_triangle(true);
        
	// Get the colouring of A
        coloring col;
      	col.compute(A);
        sparsemat B = col.permuted;
        B.extract_triangle(true);
        
        // Make sure diagonal is all ones to avoid division by zero
       
        // extract blocks once
        std::vector<sparsemat> Lblocks, Bblocks;
        col.extract_blocks(B, Lblocks, Bblocks);

        // prepare RHS
        std::vector<double> bOrig(A.n,1.0), bPerm(A.n);
        for(int i=0;i<A.n;++i) bPerm[i] = bOrig[col.perm[i]];

        // warm-up
        std::vector<double> x1, x2, x3;

        solver::mklTriSolve(B,true,bPerm,x1);

        solver::blockBiDiagSolveTasksAffinity(B,col.stagePtr,bPerm,x2);
        
        solver::kokkosSpTRSV(B, bPerm, x3);

        //time both solvers K times
        const int K = 100;
        double t_mkl=0, t_tasks=0, t_kokkos=0;
        for(int i=0;i<K;++i) {
            t_mkl   += solver::mklTriSolve(B,true,bPerm,x1);
            t_tasks += time_ms([&]{ solver::blockBiDiagSolveTasksAffinity(B,col.stagePtr,bPerm,x2); });
	    t_kokkos += solver::kokkosSpTRSV(B, bPerm, x3);
        }
        t_mkl   /= K;
        t_tasks /= K;
	t_kokkos /= K;
	
	std::cout << "Max absolute error: "<< solver::maxAbsError(x2, x3) << std::endl;

        double speedup_mkl = t_kokkos / t_mkl;
	    double speedup_tasks = t_kokkos / t_tasks;
	
        out << file << ","
            << A.n  << ","
            << A.col.size() << ","
	    << A.col.size()/A.n << ","
            << t_mkl   << ","
            << t_tasks << ","
	    << t_kokkos << ","
	    << speedup_tasks << ","
            << speedup_mkl << "\n";
        std::cout << "Done " << file << "\n";
    }

    LIKWID_MARKER_CLOSE;     // <- flush counters & write file
    
    Kokkos::finalize();

    return 0;
}



