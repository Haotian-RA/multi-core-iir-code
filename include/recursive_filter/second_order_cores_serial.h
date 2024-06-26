#ifndef SECOND_ORDER_CORES_H
#define SECOND_ORDER_CORES_H 1

#include <array>
#include "vectorclass.h"
#include "zero_init_condition_serial.h"
#include "init_cond_correction_serial.h"
#include "permuteV.h"

// different combinations of second order cores composed by zic and icc functions
template<typename V> class IirCoreOrderTwo{

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size();

    private:

        // coefficients of recursive equation: y_n = x_n + b_1x_{n-1} + b_2x_{n-2} + a_1y_{n-1} + a_2y_{n-2}
        T _b1, _b2, _a1, _a2; 

        // state for zic 
        ZeroInitCond<V> _Zic;
        
        // state for icc
        InitCondCorc<V> _Icc;
        
    public:

        // default constructor
        IirCoreOrderTwo(){};

        // Parameterized constructor, initialize the coefficients and pre-conditions of both parts with seperated values.
        IirCoreOrderTwo(const T b1, const T b2, const T a1, const T a2, const T xi1=0, const T xi2=0, const T yi1=0, const T yi2=0): 
                        _b1(b1), _b2(b2), _a1(a1), _a2(a2) {

                            // initialize the state of particular part.
                            _Zic = ZeroInitCond<V>(_b1, _b2, _a1, _a2, xi1, xi2); 

                            // initialize the state of homogeneous part.
                            _Icc = InitCondCorc<V>(_a1, _a2, yi1, yi2); 
                        };

        // Overloaded constructor, initialize the coefficients and pre-conditions of both parts with a vector of values. 
        IirCoreOrderTwo(const T coefs[5], const T inits[4]): _b1(coefs[1]), _b2(coefs[2]), _a1(coefs[3]), _a2(coefs[4]) {

            // initialize the state of particular part.
            _Zic = ZeroInitCond<V>(coefs[1], coefs[2], coefs[3], coefs[4], inits[0], inits[1]); 

            // initialize the state of homogeneous part.
            _Icc = InitCondCorc<V>(coefs[3], coefs[4], inits[2], inits[3]); 
        };



        inline void inits_refresh(const T inits[4]){

            // initialize the state of particular part.
            _Zic.inits_refresh(inits[0], inits[1]); 

            // initialize the state of homogeneous part.
            _Icc.inits_refresh(inits[2], inits[3]); 
        }
        


        /* 
        
            Second order cores for composing higher order recursive filter, which has two parts:
            1. basic second order cores:
                benchmark: accuracy check
                option1: block filtering
                option2: mixed filtering
                option3: multi-block filtering
            2. second order cores in cascaded system:
                option2_tail: the mat transpose at the head of option 2 is cancelled
                option3_head: the mat transpose at the tail of option 3 is cancelled
                option3_tail: the mat transpose at the head of option 3 is cancelled
                option3_middle: the mat transposes at head and tail of option 3 are cancelled

         */


        /* 
            basic second order cores
         */
        
        // the basic second order filter by processing scalars for accuracy check
        inline T benchmark(const T x) {

            T w = _Zic.ZIC_S(x); 
            T y = _Icc.ICC_S(w); 

            return y;
        };

        // the option 1, block filtering: ZIC_NT - ICC_NT 
        inline V option1(const V x) {

            V w = _Zic.ZIC_NT(x);
            V y = _Icc.ICC_NT(w);

            return y;
        };

        // the option 2, mixed filtering: T - ZIC_T - T - ICC_NT 
        inline std::array<V,M> option2(const std::array<V,M>& x) {

            std::array<V,M> x_T = _permuteV(x);
            std::array<V,M> w_T = _Zic.ZIC_T(x_T);
            std::array<V,M> w = _permuteV(w_T);

            std::array<V,M> y;
            for (auto n=0; n<M; n++) y[n] = _Icc.ICC_NT(w[n]);

            return y;
        };

        // the option 3, multi-block filtering: T - ZIC_T - ICC_T - T
        inline std::array<V,M> option3(const std::array<V,M>& x) {

            std::array<V,M> x_T = _permuteV(x);
            std::array<V,M> w_T = _Zic.ZIC_T(x_T);
            std::array<V,M> y_T = _Icc.ICC_T(w_T);
            std::array<V,M> y = _permuteV(y_T);

            return y;
        };

        /* 
            second order cores in cascaded system
        */

        // option 2 at the tail in cas system. The mat transpose at the head can be cancelled by another MT at the tail of previous core
        inline std::array<V,M> option2_tail(const std::array<V,M>& x_T) {

            std::array<V,M> w_T = _Zic.ZIC_T(x_T);
            std::array<V,M> w = _permuteV(w_T);

            std::array<V,M> y;
            for (auto n=0; n<M; n++) y[n] = _Icc.ICC_NT(w[n]);

            return y;
        };

        // option 3 at the head in cas system. The mat transpose at the tail can be cancelled by another MT at the head of next core
        inline std::array<V,M> option3_head(const std::array<V,M>& x) {

            std::array<V,M> x_T = _permuteV(x);
            std::array<V,M> w_T = _Zic.ZIC_T(x_T);
            std::array<V,M> y_T = _Icc.ICC_T(w_T);

            return y_T;
        };

        // option 3 at the tail in cas system. The mat transpose at the head can be cancelled by another MT at the tail of previous core
        inline std::array<V,M> option3_tail(const std::array<V,M>& x_T) {

            std::array<V,M> w_T = _Zic.ZIC_T(x_T);
            std::array<V,M> y_T = _Icc.ICC_T(w_T);
            std::array<V,M> y = _permuteV(y_T);

            return y;
        };

        // option 3 at the middle in cas system. The mat transpose at the head and tail can both be cancelled.
        inline std::array<V,M> option3_middle(const std::array<V,M>& x_T) {

            std::array<V,M> w_T = _Zic.ZIC_T(x_T);
            std::array<V,M> y_T = _Icc.ICC_T(w_T);

            return y_T;
        };

};

#endif // header guard 
