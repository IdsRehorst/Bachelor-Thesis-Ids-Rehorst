#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include "sparsemat.h"

/// A collection of solver routines for sparsemat objects.
class solver
{
public:
    /// Solve B·x = b (lower or upper triangular) via Intel MKL.
    /// Prints the solve time (excluding matrix creation).
    static void mklTriSolve(const sparsemat &B,
                            bool lower,
                            const std::vector<double> &b,
                            std::vector<double> &x);

    /// Redundant two-phase, task-based block-bidiagonal solve:
    ///  Phase 1: x′ᵢ = Lᵢ⁻¹ bᵢ           (all blocks in parallel)
    ///  Phase 2: xᵢ  = Lᵢ⁻¹ (bᵢ – Bᵢ x′ᵢ₋₁)  (all corrections in parallel)
    static void blockBiDiagSolveTasks(const sparsemat&        B,
                                      const std::vector<int>& stagePtr,
                                      const std::vector<double>& b,
                                      std::vector<double>&       x);

    /// Forward block‐bidiagonal solve by simple block‐substitution:
    ///   x1 = L1⁻¹ b1
    ///   xi = Li⁻¹ (bi − Bi xi-1),  i=2..k
    /// where blocks are given by stagePtr.
    static void blockBiDiagSolve(const sparsemat&        B,
                                 const std::vector<int>& stagePtr,
                                 const std::vector<double>& b,
                                 std::vector<double>&       x);

    /// Block‐solve by extracting each Li/Bi and calling MKL on Li
    static void blockBiDiagSolveExtract(const sparsemat&        B,
                                        const std::vector<int>& stagePtr,
                                        const std::vector<double>& b,
                                        std::vector<double>&       x);


    /// Function to determine the absolute error between two solution vectors
    static double maxAbsError(const std::vector<double>& xA,
                              const std::vector<double>& xB);

    /// Print the first `count` entries of vector `x` with a label.
    static void printFirst(const std::string &label,
                           const std::vector<double> &x,
                           size_t count = 50);


    /// Detailed error report between xA and xB
    ///  - prints max‐error index+values
    ///  - prints the top `count` errors
    static void printErrorSummary(
        const std::vector<double>& xA,
        const std::vector<double>& xB,
        size_t count = 5);

    /// A plain, row‐by‐row sparse triangular solve (SpTRSV),
    /// avoids blocks and should match MKL exactly.
    static void serialSpTRSV(const sparsemat& B,
                             bool lower,
                             const std::vector<double>& b,
                             std::vector<double>& x);
};



#endif //SOLVER_H
