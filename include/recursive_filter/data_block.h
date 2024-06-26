#ifndef DATA_BLOCK_H
#define DATA_BLOCK_H 1

#include <array>
#include "vectorclass.h"
#include <vector>

// A class of data block that encapsulates input samples, tag, initial values of sos. 
template<typename V> struct DataBlock{

    using T = decltype(std::declval<V>().extract(0));
    static constexpr int M = V::size();

    size_t tag;
    std::array<V,M> data;
    std::array<V,M> ori_data;
    std::array<T,2> x_inits; // 0: xi2, 1: xi1
    std::array<T,2> y_inits; // 0: yi2, 1: yi1
    bool last = false;       // flag of the last data block
    std::vector<T> post_inits;    // sos1: xi2, xi1, yi2, yi1, sos2: xi2, xi1, yi2, yi1
        
};

#endif // header guard 
