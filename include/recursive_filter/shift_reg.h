#ifndef SHIFT_REG_H
#define SHIFT_REG_H 1

#include "vectorclass.h"
#include <utility>

// shift register that stores the pre-conditions of recursive filter for current blocks of data 
template<typename V> class Shift{

    // V: data type of SIMD vector. T: data type of values in SIMD vector 
    using T = decltype(std::declval<V>().extract(0));

    // M: length of SIMD vector.
    constexpr static int M = V::size(); 

    private:

        // buffer initialization
        V _buffer{0};

    public:

        // scalar shift
        inline void shift(const T x) {
            // SSE
            if constexpr (M == 4) {
                // left shift buffer by 1 and add scalar at the end
                _buffer = blend4<1,2,3,4>(_buffer, x); 
            }
            
            // AVX2
            if constexpr (M == 8) {
                _buffer = blend8<1,2,3,4,5,6,7,8>(_buffer, x); 
            }
            
            // AVX512
            if constexpr (M == 16) {
                _buffer = blend16<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>(_buffer, x);
            }
        }; 
            
        // vector shift, simply copy vector in buffer thus faster than scalar shift.
        inline void shift(const V x) { 
            _buffer = x; 
        }; 

        // read data in buffer
        inline T operator[](const int idx) { 
            return (idx < 0) ? _buffer[M+idx] : _buffer[idx]; 
        }; 

};

#endif // header guard 
