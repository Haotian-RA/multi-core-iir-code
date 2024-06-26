#ifndef INIT_ADDER_H
#define INIT_ADDER_H 1

#include <array>
#include "vectorclass.h"
#include "data_block.h"
#include "shift_reg.h"

// buffer inputs and attach initials of inputs (xi1, xi2)
template<typename V> class InitAdder{

    using T = decltype(std::declval<V>().extract(0));
    static constexpr int M = V::size();

    private: 

        Shift<V> _S;

    public:

        InitAdder(const T xi1=0, const T xi2=0){

            _S.shift(xi2);
            _S.shift(xi1);
        };
    
    inline DataBlock<V> operator()(DataBlock<V> in){

        in.x_inits[0] = _S[-2];
        in.x_inits[1] = _S[-1];

        // update inits after permutation
        _S.shift(in.data[M-2][M-1]);
        _S.shift(in.data[M-1][M-1]);

        return in;
    };

};

#endif // header guard 
