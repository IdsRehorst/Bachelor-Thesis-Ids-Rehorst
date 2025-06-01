#include "coloring.h"
#include <algorithm>
#include <iostream>
#include <RACE/interface.h>

//------------------------------------------------------------------------------
// Destructor: free RACE arrays
//------------------------------------------------------------------------------
coloring::~coloring()
{
    free(perm);
    free(invPerm);
}

//------------------------------------------------------------------------------
// compute : calls RACE, permutes A, then distance‐1 level‐set
//------------------------------------------------------------------------------
void coloring::compute(sparsemat& A)
{
    // 1) call RACE to get perm / invPerm
    RACE::Interface race(A.n, /*nthreads=*/1, RACE::ONE,
                         A.rowPtr.data(), A.col.data(),
                         /*symm_hint*/false,
                         /*SMT       */1,
                         static_cast<RACE::PinMethod>(-1));

    if (race.RACEColor() != RACE_SUCCESS) {
        std::cerr << "RACEColor failed\n";
        std::exit(1);
    }
    race.getPerm(&perm,    &len);
    race.getInvPerm(&invPerm, &len);

    // 2) symmetrically permute A into 'permuted'
    permute_sym(A, perm, invPerm, permuted);

    // 3) distance‐1 level‐set on 'permuted'
    int n = permuted.n;
    level.assign(n, 0);
    int Lmax = 0;

    for (int r = 0; r < n; ++r) {
        int lv = 0;
        for (int p = permuted.rowPtr[r]; p < permuted.rowPtr[r+1]; ++p) {
            int c = permuted.col[p];
            if (c < r) lv = std::max(lv, level[c] + 1);
        }
        level[r] = lv;
        Lmax     = std::max(Lmax, lv);
    }

    stagePtr.assign(Lmax + 2, 0);
    for (int r = 0; r < n; ++r) ++stagePtr[level[r] + 1];
    for (int c = 0; c <= Lmax; ++c) stagePtr[c+1] += stagePtr[c];
}

//------------------------------------------------------------------------------
// print : show each level's row‐range, optionally mapping back via origPerm
//------------------------------------------------------------------------------
void coloring::print(const int* origPerm) const
{
    int k = static_cast<int>(stagePtr.size()) - 1;
    for (int c = 0; c < k; ++c) {
        int first = stagePtr[c],
            last  = stagePtr[c+1] - 1;
        std::cout << "Level " << c
                  << "  rows " << first << " … " << last;
        if (origPerm) {
            std::cout << "  (original indices "
                      << origPerm[first] << " … "
                      << origPerm[last] << ')';
        }
        std::cout << '\n';
    }
}

/*---------------------------------------------------------------
 *  permute – build B = P·A·Pᵀ
 *--------------------------------------------------------------*/
void coloring::permute(int               n,
                       const sparsemat&  A,
                       const int*        perm,
                       const int*        invPerm,
                       sparsemat&        B)
{
    B.n = n;
    B.rowPtr.assign(n + 1, 0);

    // count nnz per new row
    for (int newR = 0; newR < n; ++newR)
        B.rowPtr[newR + 1] =
            A.rowPtr[perm[newR] + 1] - A.rowPtr[perm[newR]];

    // prefix sum
    for (int i = 0; i < n; ++i) B.rowPtr[i + 1] += B.rowPtr[i];

    int nnz = B.rowPtr.back();
    B.col.resize(nnz);
    B.val.resize(nnz);

    std::vector<int> cursor = B.rowPtr;
    for (int newR = 0; newR < n; ++newR) {
        int oldR = perm[newR];
        for (int p = A.rowPtr[oldR]; p < A.rowPtr[oldR + 1]; ++p) {
            int q     = cursor[newR]++;
            B.col[q]  = invPerm[A.col[p]];  // relabel column
            B.val[q]  = A.val[p];
        }
    }
}

/*------------------------------------------------------------------
 *  permute_sym  –  replicate RACE's symmetric permutation
 *                 (no fancy block_size handling, just plain CSR)
 *-----------------------------------------------------------------*/
void coloring::permute_sym(const sparsemat&  A,
                           const int*        perm,
                           const int*        invPerm,
                           sparsemat&        B)
{
    const int n   = A.n;
    const int nnz = static_cast<int>(A.val.size());

    B.n = n;
    B.rowPtr.assign(n + 1, 0);

    /* count nnz per new row */
    for (int newR = 0; newR < n; ++newR)
        B.rowPtr[newR + 1] =
            A.rowPtr[perm[newR] + 1] - A.rowPtr[perm[newR]];

    /* prefix sum */
    for (int r = 0; r < n; ++r) B.rowPtr[r + 1] += B.rowPtr[r];

    B.col.resize(nnz);
    B.val.resize(nnz);
    std::vector<int> cursor = B.rowPtr;

    /* scatter + relabel columns */
    for (int newR = 0; newR < n; ++newR) {
        int oldR = perm[newR];
        for (int p = A.rowPtr[oldR]; p < A.rowPtr[oldR + 1]; ++p) {
            int q      = cursor[newR]++;
            B.col[q]   = invPerm[A.col[p]];
            B.val[q]   = A.val[p];
        }
    }

    /* -------- sort columns inside every row -------------------- */
    std::vector<std::pair<int,double>> tmp;
    for (int r = 0; r < n; ++r) {
        int begin = B.rowPtr[r];
        int end   = B.rowPtr[r+1];
        int len   = end - begin;
        if (len <= 1) continue;

        tmp.resize(len);
        for (int k = 0; k < len; ++k)
            tmp[k] = { B.col[begin+k], B.val[begin+k] };

        std::sort(tmp.begin(), tmp.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        for (int k = 0; k < len; ++k) {
            B.col[begin+k] = tmp[k].first;
            B.val[begin+k] = tmp[k].second;
        }
    }
}

void coloring::extract_blocks(const sparsemat& A,
                              std::vector<sparsemat>& Lblocks,
                              std::vector<sparsemat>& Bblocks) const
{
    int k = (int)stagePtr.size() - 1;
    Lblocks.resize(k);
    Bblocks.resize(k);

    for (int i = 0; i < k; ++i) {
        int r0 = stagePtr[i], r1 = stagePtr[i+1];
        int m  = r1 - r0;

        // build Lblocks[i] of size m×m
        sparsemat Li(m);
        Li.rowPtr.assign(m+1,0);
        // count
        for (int r = r0; r < r1; ++r)
            for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p)
                if (A.col[p] >= r0 && A.col[p] < r1)
                    ++Li.rowPtr[r-r0+1];
        // prefix
        for (int rr = 0; rr < m; ++rr)
            Li.rowPtr[rr+1] += Li.rowPtr[rr];
        // scatter
        {
            std::vector<int> cur = Li.rowPtr;
            Li.col.resize(cur.back());
            Li.val.resize(cur.back());
            for (int r = r0; r < r1; ++r)
                for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p)
                    if (int c=A.col[p]; c>=r0 && c<r1) {
                        int q = cur[r-r0]++;
                        Li.col[q] = c - r0;
                        Li.val[q] = A.val[p];
                    }
        }
        Lblocks[i] = std::move(Li);

        // build Bblocks[i] of size m×m_prev (for i>0)
        if (i>0) {
            int pr0 = stagePtr[i-1], pr1 = r0;
            int mp  = pr1-pr0;
            sparsemat Bi(m);
            Bi.rowPtr.assign(m+1,0);
            // count
            for (int r = r0; r < r1; ++r)
                for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p)
                    if (int c=A.col[p]; c>=pr0 && c<pr1)
                        ++Bi.rowPtr[r-r0+1];
            // prefix
            for (int rr = 0; rr < m; ++rr)
                Bi.rowPtr[rr+1] += Bi.rowPtr[rr];
            // scatter
            {
                std::vector<int> cur = Bi.rowPtr;
                Bi.col.resize(cur.back());
                Bi.val.resize(cur.back());
                for (int r = r0; r < r1; ++r)
                    for (int p = A.rowPtr[r]; p < A.rowPtr[r+1]; ++p)
                        if (int c=A.col[p]; c>=pr0 && c<pr1) {
                            int q = cur[r-r0]++;
                            Bi.col[q] = c - pr0;
                            Bi.val[q] = A.val[p];
                        }
            }
            Bblocks[i] = std::move(Bi);
        }
        else {
            Bblocks[i] = sparsemat(0);
        }
    }
}