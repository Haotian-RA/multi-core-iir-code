#ifndef ZERO_INIT_CONDITION_H
#define ZERO_INIT_CONDITION_H 1

#include <array>
#include "vectorclass.h"
#include "shift_reg.h"

// zero initial condition that calculates the particular part of recursive equation.
template<typename V> class ZeroInitCond{

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size(); 

    private:

        // coefficients of recursive equation: y_n = x_n + b_1x_{n-1} + b_2x_{n-2} + a_1y_{n-1} + a_2y_{n-2}
        T _b1, _b2, _a1, _a2; 

        // shift register inside zic storing the pre-condition of particular part, i.e., x_{-1}, x_{-2}.
        Shift<V> _S;

        // four vectors works for block filtering. B=[p2 p1], A=[h2 h1].
        V _p2, _p1, _h2, _h1;

        // M by M matrix works for block filtering.
        std::array<V,M> _H;

    public:

        // default constructor
        ZeroInitCond(){};

        // Parameterized constructor, initialize the particular part of recursive equation, including the coefficients and pre-conditions
        ZeroInitCond(const T b1, const T b2, const T a1, const T a2, const T xi1=0, const T xi2=0): _b1(b1), _b2(b2), _a1(a1), _a2(a2) {

            // initialize the pre-conditions of the particular part: x_{-2}, x_{-1}.
            _S.shift(xi2);
            _S.shift(xi1);

            // pre-compute matrix B and A.
            impulse_response();

            // pre-compute the transition matrix H in block filtering
            H();
        };

        inline void inits_refresh(const T xi2, const T xi1){

            _S.shift(xi2);
            _S.shift(xi1);

        }


        /* 
        
            Functions for calculating particular part of second order recursive equation, which are
            ZIC_s: scalar, sample by sample.
            ZIC_NT: block filtering.
            ZIC_T: multi-block filtering.
        
         */

        
        // calculate the particular part of recursive equation by scalar
        inline T ZIC_S(const T x) {
            T w = x + _b1*_S[-1] + _b2*_S[-2];

            _S.shift(x);

            return w;
        };

        // calculate the particular part of recursive equation by block filtering
        inline V ZIC_NT(const V x) {

            // for (int n=0;n<4;n++)
            //     for (int m=0;m<4;m++)
            //         std::cout<<_H[n][m];
            //     std::cout<<""<<std::endl;

            // for (int n=0;n<4;n++)
            //         std::cout<<_p2[n];
            //     std::cout<<""<<std::endl;
            

            V w{0};

            for (auto n=0; n<M; n++) {
                w = mul_add(_H[n], x[n], w);
            } 

            w = mul_add(_p2, _S[-2], w);
            w = mul_add(_p1, _S[-1], w);

            // vector shift: store the initial conditions for the next block of data.
            _S.shift(x);

            return w; 
        };

        // calculate the particular part of recursive equation by multi-block filtering
        inline std::array<V,M> ZIC_T(const std::array<V,M>& x) {
            std::array<V,M> v, w;

            // the two blocks contains the initial conditions in particular part
            V xi2, xi1;
            
            // SSE
            if constexpr (M == 4) {
                // get the two initial-condition blocks, xi2=[x_{-2} x_{M-2} x_{2M-2} ...], xi1=[x_{-1} x_{M-1} x_{2M-1} ...]
                xi2 = blend4<4,0,1,2>(x[M-2], _S[-2]);
                xi1 = blend4<4,0,1,2>(x[M-1], _S[-1]);
            }

            // AVX2
            if constexpr (M == 8) {
                xi2 = blend8<8,0,1,2,3,4,5,6>(x[M-2], _S[-2]);
                xi1 = blend8<8,0,1,2,3,4,5,6>(x[M-1], _S[-1]);
            }

            // AVX512
            if constexpr (M == 16) {
                xi2 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x[M-2], _S[-2]);
                xi1 = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(x[M-1], _S[-1]);
            }

            /* 
                Perform computation of zic:
                interleave the computation by non-dependency part (multiply by b1 and b2) and dependency (a1 and a2)
                to reduce the waiting time of read-after-write (dependency) issue. Note, this can be automatically done 
                by using newer version of compiler and faster compiling flags, e.g., -O2, -O3.
             */
            v[0] = mul_add(xi2, _b2, x[0]);
            v[0] = mul_add(xi1, _b1, v[0]);
            w[0] = v[0];
            v[1] = mul_add(xi1, _b2, x[1]);
            v[1] = mul_add(x[0], _b1, v[1]);
            w[1] = mul_add(v[0], _a1, v[1]);

            for (auto n=2; n<M; n++) {
                v[n] = mul_add(x[n-2], _b2, x[n]);
                v[n] = mul_add(x[n-1], _b1, v[n]);
                w[n] = mul_add(w[n-2], _a2, v[n]);
                w[n] = mul_add(w[n-1], _a1, w[n]);
            }

            /* 
                2 times scalar shift:
                store initial conditions for the next block of data, which are the last samples in the last two blocks of X^T.
             */
            _S.shift(x[M-2][M-1]);
            _S.shift(x[M-1][M-1]);

            return w; 
        };


        /* 
        
            Pre-computations in each function.
        
         */


        // calculate matrix B and A for block filtering. The addition of b_1 and a_1 is the lagged impulse response of recursive equation. 
        inline void impulse_response() {
            T p2[M+1], p1[M+1], h0[M+1];

            p2[0] = _b2;
            p2[1] = _a1*_b2;
            p1[0] = _b1;
            p1[1] = _a1*_b1 + _b2;
            h0[0] = 1;
            h0[1] = _a1;

            // std::cout<<p2[0]<<std::endl;

            for (auto n=2; n<M+1; n++){
                p2[n] = _a1*p2[n-1] + _a2*p2[n-2];
                // std::cout<<p2[2]<<std::endl;
                p1[n] = _a1*p1[n-1] + _a2*p1[n-2];
                h0[n] = _a1*h0[n-1] + _a2*h0[n-2];  
            }

            // std::cout<<p2[1]<<std::endl;
            // for (int n=0;n<4;n++)
            //     std::cout<<p2[n];
            // std::cout<<""<<std::endl;

            _p2.load(&p2[0]);
            _p1.load(&p1[0]);
            _h2.load(&h0[0]);
            _h2 *= _a2;
            _h1.load(&h0[1]);

            // for (int n=0;n<4;n++)
            //     std::cout<<p2[n];
            // std::cout<<""<<std::endl;

            // std::cout<<p2[0]<<std::endl;
        };

        // calculate the transition matrix H for block filtering, which is a lower triangular toplitz matrix.
        inline void H() {
            V tmp;
            
            // SSE
            if constexpr (M == 4) {
                // calcualte the first column in H (the exact impulse response), which can be obtained inversely by the addition of h1 and p1. 
                tmp = blend4<4,0,1,2>(_h1+_p1, 1);

                for (auto n=0; n<M; n++){
                    _H[n] = tmp;
                    // the rest columns can be shifted from the first column by 1 position in H 
                    tmp = permute4<-1,0,1,2>(tmp);
                }
            }

            // AVX2
            if constexpr (M == 8) { 
                tmp = blend8<8,0,1,2,3,4,5,6>(_h1+_p1, 1);

                for (auto n=0; n<M; n++){
                    _H[n] = tmp;
                    tmp = permute8<-1,0,1,2,3,4,5,6>(tmp);
                }
            }

            // AVX512
            if constexpr (M == 16) { 
                tmp = blend16<16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(_h1+_p1, 1);

                for (auto n=0; n<M; n++){
                    _H[n] = tmp;
                    tmp = permute16<-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>(tmp);
                }
            }
        };
        
};

#endif // header guard 
