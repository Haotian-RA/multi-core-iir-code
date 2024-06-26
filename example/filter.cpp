#include "recursive_filter.h"
#include <tbb/tbb.h>
#include <chrono>
#include <iostream>

// AVX2
using V = Vec8f;

// V: data type of SIMD vector. T: data type of values in SIMD vector 
using T = decltype(std::declval<V>().extract(0));

// M: length of SIMD vector.
constexpr static int M = V::size();

int main(){

    // filter parameters:
    constexpr int L = 12; // order of filter
    // array of coefficients and initial conditions
    T inits[L/2][4] = {0.2,0.3,0.4,0.5
                      ,0.5,0.7,0.9,3
                      ,-2,2.1,3,9
                      ,2.2,2.3,0.5,0.3
                      ,-2,-3,5,7
                      ,2,3,1,8
                      };
    T coefs[L/2][5] = {-0.5,0.25,-0.75,0.6
                      ,0.5,0.7,0.9,0.1
                      ,-0.2,0.2,0.3,0.9
                      ,-0.4,0.5,0.5,0.1
                      ,-0.25,-0.3,0.15,0.7
                      ,0.12,0.23,0.31,0.8
                      };

    // 1.024M samples
    static const int vector_size = 1024000;
    // input: an equal-size impulse response
    std::array<T,vector_size> in = {0}, out;
    in[0] = 1;

    auto multi_core_filter = makeMultiCoreFilter(coefs,inits);

    auto start = std::chrono::high_resolution_clock::now();

    multi_core_filter(in.begin(),in.end(),out.begin());

    auto finish = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

    return 0;

}
