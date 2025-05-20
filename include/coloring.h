#ifndef COLORING_H
#define COLORING_H

#include "sparsemat.h"
#include <vector>

/*---------------------------------------------------------------
 * Very small helper class
 *  • compute distance-1 levels   (compute())
 *  • print them                  (print())
 *  • build a permuted copy       (permute())
 *--------------------------------------------------------------*/
class coloring
{
public:
    // Raw permutation arrays from RACE
    int  *perm     = nullptr;  // newRow → oldRow
    int  *invPerm  = nullptr;  // oldRow → newRow
    int   len      = 0;        // length of perm/invPerm = n

    // Permuted matrix
    sparsemat permuted;

    // Level‐set output
    std::vector<int> level;     // level[i]  = colour of row i in permuted
    std::vector<int> stagePtr;  // stagePtr[c] = first row of colour c

    // Compute perm, permuted matrix, level & stagePtr
    // from original matrix A
    void compute(sparsemat& A);

    // Print levels (in permuted index space), optional original mapping
    void print(const int* origPerm = nullptr) const;

    // Cleanup RACE‐allocated arrays
    ~coloring();

    // Symmetric CSR permutation: B = P·A·Pᵀ
    static void permute_sym(const sparsemat& A,
                            const int*       perm,
                            const int*       invPerm,
                            sparsemat&       B);

    // Extract blocks from permuted
    void extract_blocks(const sparsemat& A,
                        std::vector<sparsemat>& Lblocks,
                        std::vector<sparsemat>& Bblocks) const;

    /*---- build B = P·A·P^T ---------------------------------- */
    static void permute(int n,
                        const sparsemat& A,
                        const int* perm, // new→old
                        const int* invPerm,
                        sparsemat& B); // old→new
};



#endif //COLORING_H
