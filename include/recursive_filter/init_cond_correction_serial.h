#ifndef INIT_COND_CORRECTION_H
#define INIT_COND_CORRECTION_H 1

#include <array>
#include "vectorclass.h"
#include "shift_reg.h"

// initial condition correction that calculates the homogeneous part of recursive equation.
template<typename V> class InitCondCorc{

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    private:

        // coefficients of recursive equation: y_n = x_n + b_1x_{n-1} + b_2x_{n-2} + a_1y_{n-1} + a_2y_{n-2}
        T _a1, _a2; 
        
        // vectors contain the elements at the four positions of C, C^2, C^3 ...
        V _h_22, _h_12, _h_21, _h_11;

        // pre-compute the vectors including C for recursive doubling initialization
        V _rd0_22, _rd0_12, _rd0_21, _rd0_11; 

        // pre-compute the vectors including C for recursive doubling post correction
        V _rdb_22, _rdb_12, _rdb_21, _rdb_11; 

        // pre-compute the vectors including C for RD first recursion
        V _rd1_22, _rd1_12, _rd1_21, _rd1_11; 

        // pre-compute the vectors including C for RD second recursion
        V _rd2_22, _rd2_12, _rd2_21, _rd2_11;

        // pre-compute the vectors including C for RD third recursion
        V _rd3_22, _rd3_12, _rd3_21, _rd3_11;

        // pre-compute the vectors including C for RD fourth recursion
        V _rd4_22, _rd4_12, _rd4_21, _rd4_11;

        // the first vectors for large matrix T for old large matrix multiplication (MM) method
        std::array<V,M> _T_22, _T_12, _T_21, _T_11;

        // shift register inside icc storeing the pre-condition of homogeneous part, i.e., y_{-1}, y_{-2}.
        Shift<V> _S;

        // vectors in matrix A, A=[h2 h1].
        V _h2, _h1;

        int tag = -1;

    public:

        // default constructor
        InitCondCorc(){};

        // Parameterized constructor, initialize the homogeneous part of recursive equation, including the coefficients and pre-conditions
        InitCondCorc(const T a1, const T a2, const T yi1=0, const T yi2=0): _a1(a1), _a2(a2) { 

            // initialize the pre-conditions of the homogeneous part: y_{-2}, y_{-1}.
            _S.shift(yi2);
            _S.shift(yi1);

            // pre-compute matrix A.
            impulse_response();

            // pre-compute the vectors including C in recursive doubling.
            recursive_doubling_vectors();

            // pre-compute matrix T(and D) in matrix multplication (MM) method
        };

        inline void inits_refresh(const T yi2, const T yi1){

            _S.shift(yi2);
            _S.shift(yi1);

        }
        
        /* 
        
            Functions for calculating homogeneous part of second order recursive equation, which are
            ICC_NT: block filtering
            ICC_T: multi-block filtering by recursive filtering (in the paper, recommand)
            ICC_T_MM: multi-block filtering by matrix multiplication (in the paper, not recommand)
            ICC2_T: multi-block filtering by recursive filtering in a different tree (not in the paper, slower, not recommand)

         */


        // calculate the homogeneous part of recursive equation by scalar
        inline T ICC_S(const T w) {
            T y = w + _a1*_S[-1] + _a2*_S[-2];

            _S.shift(y);

            return y;
        };

        // calculate the homogeneous part of recursive equation by block filtering
        inline V ICC_NT(const V w) {
            V y;

            y = mul_add(_h2, _S[-2], w);
            y = mul_add(_h1, _S[-1], y);

            // vector shift: store the initial conditions for the next block of data.
            _S.shift(y);

            return y;
        };

        // prior correction: calculate the homogeneous part of recursive equation by multi-block filtering and recursive doubling.
        inline std::array<V,M> ICC_T(const std::array<V,M>& w) { 
            std::array<V,M> y;

            // the two blocks contains the initial conditions in homogeneous part, Y_p^T=[yi2 yi1].
            V yi2, yi1;
            
            // b2 and b1 are not coefficients in icc as in zic, but the temporary vectors working for RD. 
            V b2, b1;

            // recursive doubling step 1: initialization
            y[M-2] = mul_add(_rd0_22, _S[-2], w[M-2]);
            y[M-2] = mul_add(_rd0_12, _S[-1], y[M-2]);
            y[M-1] = mul_add(_rd0_21, _S[-2], w[M-1]);
            y[M-1] = mul_add(_rd0_11, _S[-1], y[M-1]);
            
            // SSE
            if constexpr (M == 4) {
                // step 2: first recursion
                b2 = permute4<-1,0,-1,2>(y[M-2]);
                b1 = permute4<-1,0,-1,2>(y[M-1]);

                y[M-2] = mul_add(b2, _rd1_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd1_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd1_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd1_11, y[M-1]);

                // step 3: second recursion
                b2 = permute4<-1,-1,1,1>(y[M-2]);
                b1 = permute4<-1,-1,1,1>(y[M-1]);

                y[M-2] = mul_add(b2, _rd2_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd2_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd2_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd2_11, y[M-1]);

                // shuffle for getting Y_p^T from the last two blocks of Y^T, i.e., Y^T_{[M-2]}, Y^T_{[M-1]}.
                yi2 = blend4<4,0,1,2>(y[M-2], _S[-2]);
                yi1 = blend4<4,0,1,2>(y[M-1], _S[-1]);
            };

            // AVX2
            if constexpr (M == 8) { 
                // step 2: first recursion
                b2 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-2]);
                b1 = permute8<-1,0,-1,2,-1,4,-1,6>(y[M-1]);

                y[M-2] = mul_add(b2, _rd1_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd1_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd1_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd1_11, y[M-1]);

                // step 3: second recursion
                b2 = permute8<-1,-1,1,1,-1,-1,5,5>(y[M-2]);
                b1 = permute8<-1,-1,1,1,-1,-1,5,5>(y[M-1]);

                y[M-2] = mul_add(b2, _rd2_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd2_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd2_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd2_11, y[M-1]);

                // step 4: third recursion
                b2 = permute8<-1,-1,-1,-1,3,3,3,3>(y[M-2]);
                b1 = permute8<-1,-1,-1,-1,3,3,3,3>(y[M-1]);

                y[M-2] = mul_add(b2, _rd3_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd3_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd3_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd3_11, y[M-1]);

                yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S[-2]);
                yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S[-1]);
            };

            // AVX512
            if constexpr (M == 16) { 
                // step 2: first recursion
                b2 = permute16<-1,0,-1,2,-1,4,-1,6,-1,8,-1,10,-1,12,-1,14>(y[M-2]);
                b1 = permute16<-1,0,-1,2,-1,4,-1,6,-1,8,-1,10,-1,12,-1,14>(y[M-1]);

                y[M-2] = mul_add(b2, _rd1_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd1_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd1_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd1_11, y[M-1]);

                // step 3: second recursion
                b2 = permute16<-1,-1,1,1,-1,-1,5,5,-1,-1,9,9,-1,-1,13,13>(y[M-2]);
                b1 = permute16<-1,-1,1,1,-1,-1,5,5,-1,-1,9,9,-1,-1,13,13>(y[M-1]);

                y[M-2] = mul_add(b2, _rd2_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd2_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd2_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd2_11, y[M-1]);

                // step 4: third recursion
                b2 = permute16<-1,-1,-1,-1,3,3,3,3,-1,-1,-1,-1,11,11,11,11>(y[M-2]);
                b1 = permute16<-1,-1,-1,-1,3,3,3,3,-1,-1,-1,-1,11,11,11,11>(y[M-1]);

                y[M-2] = mul_add(b2, _rd3_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd3_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd3_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd3_11, y[M-1]);

                // step 4: fourth recursion
                b2 = permute16<-1,-1,-1,-1,-1,-1,-1,-1,7,7,7,7,7,7,7,7>(y[M-2]);
                b1 = permute16<-1,-1,-1,-1,-1,-1,-1,-1,7,7,7,7,7,7,7,7>(y[M-1]);

                y[M-2] = mul_add(b2, _rd4_22, y[M-2]);
                y[M-2] = mul_add(b1, _rd4_12, y[M-2]);
                y[M-1] = mul_add(b2, _rd4_21, y[M-1]);
                y[M-1] = mul_add(b1, _rd4_11, y[M-1]);

                yi2 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(y[M-2], _S[-2]);
                yi1 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(y[M-1], _S[-1]);
            };

            // forward the first M-2 blocks in Y^T
            for (auto n=0; n<M-2; n++) {
                y[n] = mul_add(yi2, _h2[n], w[n]);
                y[n] = mul_add(yi1, _h1[n], y[n]);
            };
     
            /* 
                2 times scalar shift:
                store initial conditions for the next block of data, which are the last samples in the last two blocks of X^T.
             */         
            _S.shift(y[M-2][M-1]);
            _S.shift(y[M-1][M-1]); 
            
            return y; 
        };

        // post correction: calculate the homogeneous part of recursive equation by multi-block filtering and recursive doubling.
        inline std::array<V,M> ICC_T2(const std::array<V,M>& w) { 
            std::array<V,M> v, y;

            // the two blocks contains the initial conditions in homogeneous part, Y_p^T=[yi2 yi1].
            V yi2, yi1;
            
            // b2 and b1 are not coefficients in icc as in zic, but the temporary vectors working for RD. 
            V b2, b1;    

            tag++;

            // std::cout<<"*************************"<<std::endl;
            // std::cout<<"tag: "<<tag<<std::endl;
            // std::cout<<"yi1: "<<_S[-1]<<std::endl;
            // std::cout<<"yi2: "<<_S[-2]<<std::endl;

            // AVX2
            if constexpr (M == 8) { 
                // step 2: first recursion
                b2 = permute8<-1,0,-1,2,-1,4,-1,6>(w[M-2]);
                b1 = permute8<-1,0,-1,2,-1,4,-1,6>(w[M-1]);

                v[M-2] = mul_add(b2, _rd1_22, w[M-2]);
                v[M-2] = mul_add(b1, _rd1_12, v[M-2]);
                v[M-1] = mul_add(b2, _rd1_21, w[M-1]);
                v[M-1] = mul_add(b1, _rd1_11, v[M-1]);

                // step 3: second recursion
                b2 = permute8<-1,-1,1,1,-1,-1,5,5>(v[M-2]);
                b1 = permute8<-1,-1,1,1,-1,-1,5,5>(v[M-1]);

                v[M-2] = mul_add(b2, _rd2_22, v[M-2]);
                v[M-2] = mul_add(b1, _rd2_12, v[M-2]);
                v[M-1] = mul_add(b2, _rd2_21, v[M-1]);
                v[M-1] = mul_add(b1, _rd2_11, v[M-1]);

                // step 4: third recursion
                b2 = permute8<-1,-1,-1,-1,3,3,3,3>(v[M-2]);
                b1 = permute8<-1,-1,-1,-1,3,3,3,3>(v[M-1]);

                v[M-2] = mul_add(b2, _rd3_22, v[M-2]);
                v[M-2] = mul_add(b1, _rd3_12, v[M-2]);
                v[M-1] = mul_add(b2, _rd3_21, v[M-1]);
                v[M-1] = mul_add(b1, _rd3_11, v[M-1]);

            };

            // recursive doubling backward correction
            y[M-2] = mul_add(_h_22, _S[-2], v[M-2]);
            y[M-2] = mul_add(_h_12, _S[-1], y[M-2]);
            y[M-1] = mul_add(_h_21, _S[-2], v[M-1]);
            y[M-1] = mul_add(_h_11, _S[-1], y[M-1]);

            if constexpr (M == 8) { 

                yi2 = blend8<8,0,1,2,3,4,5,6>(y[M-2], _S[-2]);
                yi1 = blend8<8,0,1,2,3,4,5,6>(y[M-1], _S[-1]);

            };

            // forward the first M-2 blocks in Y^T
            for (auto n=0; n<M-2; n++) {
                y[n] = mul_add(yi2, _h2[n], w[n]);
                y[n] = mul_add(yi1, _h1[n], y[n]);
            };
     
            /* 
                2 times scalar shift:
                store initial conditions for the next block of data, which are the last samples in the last two blocks of X^T.
             */         
            _S.shift(y[M-2][M-1]);
            _S.shift(y[M-1][M-1]); 
            
            return y; 
        };

        /* 
        
            Pre-computations in each function.
        
         */


        // calculate matrix A for icc, which is further used for pre-computing vectors for recursive doubling and large MM method   
        inline void impulse_response() {

            T h0[M+1];
            
            h0[0] = 1;
            h0[1] = _a1;

            for (auto n=2; n<M+1; n++) {
                h0[n] = _a1*h0[n-1] + _a2*h0[n-2];
            }    

            _h2.load(&h0[0]);
            _h2 *= _a2;
            _h1.load(&h0[1]);
        };

        // calculate vectors contain the elements at the four positions of C, C^2, C^3 ...
        inline void C_power() { 
        
            T h_22[M] = {0}, h_12[M] = {0}, h_21[M] = {0}, h_11[M] = {0}; 

            h_22[0] = _h2[M-2];
            h_12[0] = _h1[M-2];
            h_21[0] = _h2[M-1];
            h_11[0] = _h1[M-1];

            for (auto n=1; n<M; n++) {
                h_22[n] = _h2[M-2]*h_22[n-1] + _h2[M-1]*h_12[n-1];
                h_12[n] = _h1[M-2]*h_22[n-1] + _h1[M-1]*h_12[n-1];
                h_21[n] = _h2[M-2]*h_21[n-1] + _h2[M-1]*h_11[n-1];
                h_11[n] = _h1[M-2]*h_21[n-1] + _h1[M-1]*h_11[n-1];
            }

            _h_22.load(&h_22[0]);
            _h_12.load(&h_12[0]);
            _h_21.load(&h_21[0]);
            _h_11.load(&h_11[0]);


        };

        // calculate the vectors including elements of C in recursive doubling
        inline void recursive_doubling_vectors() {

            C_power();

            // std::cout<<"icc"<<std::endl;
            // for (int i=0;i<8;i++)
            //     std::cout<<_h_22[i]<<std::endl;

            // AVX2
            if constexpr (M == 8) {

                // RD initialization, [C 0 0 0 0 0 0 0]
                _rd0_22 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_22); 
                _rd0_12 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_12);
                _rd0_21 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_21);
                _rd0_11 = permute8<0,-1,-1,-1,-1,-1,-1,-1>(_h_11);

                // RD backward, [C C^2 C^3 C^4 C^5 C^6 C^7 C^8]
                _rdb_22 = _h_22; 
                _rdb_12 = _h_12;
                _rdb_21 = _h_21;
                _rdb_11 = _h_11;

                // RD recursion 1, [0 C 0 C 0 C 0 C]
                _rd1_22 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_22);
                _rd1_12 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_12);
                _rd1_21 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_21);
                _rd1_11 = permute8<-1,0,-1,0,-1,0,-1,0>(_h_11);

                // RD recursion 2, [0 0 C C^2 0 0 C C^2]
                _rd2_22 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_22);
                _rd2_12 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_12);
                _rd2_21 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_21);
                _rd2_11 = permute8<-1,-1,0,1,-1,-1,0,1>(_h_11);

                // RD recursion 3, [0 0 0 0 C C^2 C^3 C^4]
                _rd3_22 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_22);
                _rd3_12 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_12);
                _rd3_21 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_21);
                _rd3_11 = permute8<-1,-1,-1,-1,0,1,2,3>(_h_11);
            };
        };

        /* 
            calculate matrix T(and D) in matrix multplication (MM) method.
            Note, the two matrix are not explained in details in the paper due to no efficiency.
            Basically, T is a 2M by 2M matrix, where each sub-matrix of size M by M 
            is lower triangular toplitz matrix related to 4 vectors in C power and D is exactly C power. 
         */

};

#endif // header guard 










