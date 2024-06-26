#ifndef NO_STATE_ZIC_H
#define NO_STATE_ZIC_H 1

#include <array>
#include "vectorclass.h"
#include "data_block.h"

// Stateless zero initial condition that computes the particular part of recursive equation.
template<typename V> class NoStateZIC{

    using T = decltype(std::declval<V>().extract(0));
    constexpr static int M = V::size(); 

    private:

        // coefficients of recursive equation: y_n = x_n + b_1*x_{n-1} + b_2*x_{n-2} + a_1*y_{n-1} + a_2*y_{n-2}
        T _b1, _b2, _a1, _a2; 

        // impulse response vectors
        V _p2, _p1, _h2, _h1;

    public:

        NoStateZIC(const T b1, const T b2, const T a1, const T a2, const T xi1=0, const T xi2=0): _b1(b1), _b2(b2), _a1(a1), _a2(a2) {

            // pre-compute matrix B and A.
            impulse_response();

        };

        // multi-block filtering that accepts transposed matrix of samples
        inline DataBlock<V> operator()(DataBlock<V> in) {

            if (in.last){

                in.post_inits.push_back(in.data[M-2][M-1]);
                in.post_inits.push_back(in.data[M-1][M-1]);
            };

            std::array<V,M> v, w;

            V xi2 = blend8<8,0,1,2,3,4,5,6>(in.data[M-2], in.x_inits[0]);
            V xi1 = blend8<8,0,1,2,3,4,5,6>(in.data[M-1], in.x_inits[1]);
            
            v[0] = mul_add(xi2, _b2, in.data[0]);
            v[0] = mul_add(xi1, _b1, v[0]);
            w[0] = v[0];
            v[1] = mul_add(xi1, _b2, in.data[1]);
            v[1] = mul_add(in.data[0], _b1, v[1]);
            w[1] = mul_add(v[0], _a1, v[1]);

            for (auto n=2; n<M; n++) {

                v[n] = mul_add(in.data[n-2], _b2, in.data[n]);
                v[n] = mul_add(in.data[n-1], _b1, v[n]);
                w[n] = mul_add(w[n-2], _a2, v[n]);
                w[n] = mul_add(w[n-1], _a1, w[n]);
            }

            in.data = w;

            return in; 
        };

        // calculate matrix B and A (see paper). The addition of b_1 and a_1 is the lagged impulse response of recursive equation. 
        inline void impulse_response() {

            T p2[M], p1[M], h0[M+1];

            p2[0] = _b2;
            p2[1] = _a1*_b2;
            p1[0] = _b1;
            p1[1] = _a1*_b1 + _b2;
            h0[0] = 1;
            h0[1] = _a1;

            for (auto n=2; n<M+1; n++){
                p2[n] = _a1*p2[n-1] + _a2*p2[n-2];
                p1[n] = _a1*p1[n-1] + _a2*p1[n-2];
                h0[n] = _a1*h0[n-1] + _a2*h0[n-2];  
            }

            _p2.load(&p2[0]);
            _p1.load(&p1[0]);
            _h2.load(&h0[0]);
            _h2 *= _a2;
            _h1.load(&h0[1]);
        };
        
};

#endif // header guard 
