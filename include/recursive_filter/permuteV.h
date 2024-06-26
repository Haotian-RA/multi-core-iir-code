#ifndef PERMUTEV_H
#define PERMUTEV_H 1

#include <array>
#include "vectorclass.h"

// matrix transpose for different size of matrices
template<typename V> inline std::array<V,V::size()> _permuteV(const std::array<V,V::size()>& matrix) {
    std::array<V,V::size()> matrix_T;
    // SSE
    if constexpr (V::size() == 4) _permuteV4(matrix.data(), matrix_T.data());
    // AVX2
    if constexpr (V::size() == 8) _permuteV8(matrix.data(), matrix_T.data());
    // AVX512
    if constexpr (V::size() == 16) _permuteV16(matrix.data(), matrix_T.data());

    return matrix_T;
};

// matrix transpose for matrix in size 4 by 4
template<typename V> inline void _permuteV4(const V matrix[4], V matrix_T[4]) {
    V tmp[4];

    // first round: swap lower left and upper right
    tmp[0] = blend4<0,1,4,5>(matrix[0], matrix[2]);
    tmp[1] = blend4<0,1,4,5>(matrix[1], matrix[3]);
    tmp[2] = blend4<2,3,6,7>(matrix[0], matrix[2]);
    tmp[3] = blend4<2,3,6,7>(matrix[1], matrix[3]);

    // second/final round: swap lower left and upper right within each quadrant
    matrix_T[0] = blend4<0,4,2,6>(tmp[0], tmp[1]);
    matrix_T[1] = blend4<1,5,3,7>(tmp[0], tmp[1]);
    matrix_T[2] = blend4<0,4,2,6>(tmp[2], tmp[3]);
    matrix_T[3] = blend4<1,5,3,7>(tmp[2], tmp[3]);
};

// matrix transpose for matrix in size 8 by 8
template<typename V> inline void _permuteV8(const V matrix[8], V matrix_T[8]) {
    V tmp1[8], tmp2[8];

    // first round: swap lower left and upper right
    tmp1[0] = blend8<0,1,2,3,8,9,10,11>(matrix[0], matrix[4]);
    tmp1[1] = blend8<0,1,2,3,8,9,10,11>(matrix[1], matrix[5]);
    tmp1[2] = blend8<0,1,2,3,8,9,10,11>(matrix[2], matrix[6]);
    tmp1[3] = blend8<0,1,2,3,8,9,10,11>(matrix[3], matrix[7]);
    tmp1[4] = blend8<4,5,6,7,12,13,14,15>(matrix[0], matrix[4]);
    tmp1[5] = blend8<4,5,6,7,12,13,14,15>(matrix[1], matrix[5]);
    tmp1[6] = blend8<4,5,6,7,12,13,14,15>(matrix[2], matrix[6]);
    tmp1[7] = blend8<4,5,6,7,12,13,14,15>(matrix[3], matrix[7]);

    // second/final round: swap lower left and upper right within each quadrant
    tmp2[0] = blend8<0,1,8,9,4,5,12,13>(tmp1[0], tmp1[2]);
    tmp2[1] = blend8<0,1,8,9,4,5,12,13>(tmp1[1], tmp1[3]);
    tmp2[2] = blend8<2,3,10,11,6,7,14,15>(tmp1[0], tmp1[2]);
    tmp2[3] = blend8<2,3,10,11,6,7,14,15>(tmp1[1], tmp1[3]);
    tmp2[4] = blend8<0,1,8,9,4,5,12,13>(tmp1[4], tmp1[6]);
    tmp2[5] = blend8<0,1,8,9,4,5,12,13>(tmp1[5], tmp1[7]);
    tmp2[6] = blend8<2,3,10,11,6,7,14,15>(tmp1[4], tmp1[6]);
    tmp2[7] = blend8<2,3,10,11,6,7,14,15>(tmp1[5], tmp1[7]);

    // third/final round: swap elements anti-diagonally adjacent
    matrix_T[0] = blend8<0,8,2,10,4,12,6,14>(tmp2[0], tmp2[1]);
    matrix_T[1] = blend8<1,9,3,11,5,13,7,15>(tmp2[0], tmp2[1]);
    matrix_T[2] = blend8<0,8,2,10,4,12,6,14>(tmp2[2], tmp2[3]);
    matrix_T[3] = blend8<1,9,3,11,5,13,7,15>(tmp2[2], tmp2[3]);
    matrix_T[4] = blend8<0,8,2,10,4,12,6,14>(tmp2[4], tmp2[5]);
    matrix_T[5] = blend8<1,9,3,11,5,13,7,15>(tmp2[4], tmp2[5]);
    matrix_T[6] = blend8<0,8,2,10,4,12,6,14>(tmp2[6], tmp2[7]);
    matrix_T[7] = blend8<1,9,3,11,5,13,7,15>(tmp2[6], tmp2[7]); 
};

// matrix transpose for matrix in size 16 by 16
template<typename V> inline void _permuteV16(const V matrix[16], V matrix_T[16]) {
    V tmp1[16], tmp2[16], tmp3[16];

    // first round: swap lower left and upper right
    tmp1[0] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[0], matrix[8]);
    tmp1[1] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[1], matrix[9]);
    tmp1[2] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[2], matrix[10]);
    tmp1[3] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[3], matrix[11]);
    tmp1[4] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[4], matrix[12]);
    tmp1[5] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[5], matrix[13]);
    tmp1[6] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[6], matrix[14]);
    tmp1[7] = blend16<0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23>(matrix[7], matrix[15]);
    tmp1[8] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[0], matrix[8]);
    tmp1[9] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[1], matrix[9]);
    tmp1[10] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[2], matrix[10]);
    tmp1[11] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[3], matrix[11]);
    tmp1[12] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[4], matrix[12]);
    tmp1[13] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[5], matrix[13]);
    tmp1[14] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[6], matrix[14]);
    tmp1[15] = blend16<8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31>(matrix[7], matrix[15]);

    // second/final round: swap lower left and upper right within each quadrant
    tmp2[0] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[0], tmp1[4]);
    tmp2[1] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[1], tmp1[5]);
    tmp2[2] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[2], tmp1[6]);
    tmp2[3] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[3], tmp1[7]);
    tmp2[4] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[0], tmp1[4]);
    tmp2[5] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[1], tmp1[5]);
    tmp2[6] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[2], tmp1[6]);
    tmp2[7] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[3], tmp1[7]);
    tmp2[8] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[8], tmp1[12]);
    tmp2[9] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[9], tmp1[13]);
    tmp2[10] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[10], tmp1[14]);
    tmp2[11] = blend16<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(tmp1[11], tmp1[15]);
    tmp2[12] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[8], tmp1[12]);
    tmp2[13] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[9], tmp1[13]);
    tmp2[14] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[10], tmp1[14]);
    tmp2[15] = blend16<4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31>(tmp1[11], tmp1[15]);

    // third/final round: swap lower left and upper right within each sub quadrant
    tmp3[0] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[0], tmp2[2]);
    tmp3[1] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[1], tmp2[3]);
    tmp3[2] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[0], tmp2[2]);
    tmp3[3] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[1], tmp2[3]);
    tmp3[4] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[4], tmp2[6]);
    tmp3[5] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[5], tmp2[7]);
    tmp3[6] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[4], tmp2[6]);
    tmp3[7] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[5], tmp2[7]);
    tmp3[8] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[8], tmp2[10]);
    tmp3[9] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[9], tmp2[11]);
    tmp3[10] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[8], tmp2[10]);
    tmp3[11] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[9], tmp2[11]);
    tmp3[12] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[12], tmp2[14]);
    tmp3[13] = blend16<0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29>(tmp2[13], tmp2[15]);
    tmp3[14] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[12], tmp2[14]);
    tmp3[15] = blend16<2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31>(tmp2[13], tmp2[15]);

    // third/final round: swap elements anti-diagonally adjacent
    matrix_T[0] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[0], tmp3[1]);
    matrix_T[1] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[0], tmp3[1]);
    matrix_T[2] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[2], tmp3[3]);
    matrix_T[3] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[2], tmp3[3]);
    matrix_T[4] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[4], tmp3[5]);
    matrix_T[5] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[4], tmp3[5]);
    matrix_T[6] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[6], tmp3[7]);
    matrix_T[7] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[6], tmp3[7]); 
    matrix_T[8] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[8], tmp3[9]);
    matrix_T[9] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[8], tmp3[9]);
    matrix_T[10] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[10], tmp3[11]);
    matrix_T[11] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[10], tmp3[11]);
    matrix_T[12] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[12], tmp3[13]);
    matrix_T[13] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[12], tmp3[13]);
    matrix_T[14] = blend16<0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30>(tmp3[14], tmp3[15]);
    matrix_T[15] = blend16<1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31>(tmp3[14], tmp3[15]); 
};

#endif
