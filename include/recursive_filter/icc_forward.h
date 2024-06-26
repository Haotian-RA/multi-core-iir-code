#ifndef ICC_FORWARD_H
#define ICC_FORWARD_H 1

#include "vectorclass.h"
#include "data_block.h"

// (stateless) forward the first M-2 vectors of each data block 
template<typename V> class ICCForward{

    using T = decltype(std::declval<V>().extract(0));
    constexpr static int M = V::size();

    private: 

        T _a1, _a2;
        V _h_22, _h_12, _h_21, _h_11;
        V _h2, _h1;

    public:

        ICCForward(const T a1, const T a2): _a1(a1), _a2(a2) { 

            // pre-compute matrix A.
            impulse_response();

            // pre-compute matrix A.
            C_power();

        };

        inline DataBlock<V> operator()(DataBlock<V> in){

            // recursive doubling backward correction
            in.data[M-2] = mul_add(_h_22, in.y_inits[0], in.data[M-2]);
            in.data[M-2] = mul_add(_h_12, in.y_inits[1], in.data[M-2]);
            in.data[M-1] = mul_add(_h_21, in.y_inits[0], in.data[M-1]);
            in.data[M-1] = mul_add(_h_11, in.y_inits[1], in.data[M-1]);

            V yi2 = blend8<8,0,1,2,3,4,5,6>(in.data[M-2], in.y_inits[0]);
            V yi1 = blend8<8,0,1,2,3,4,5,6>(in.data[M-1], in.y_inits[1]);

            // forward the first M-2 blocks in Y^T
            for (auto n=0; n<M-2; n++) {

                in.data[n] = mul_add(yi2, _h2[n], in.data[n]);
                in.data[n] = mul_add(yi1, _h1[n], in.data[n]);
            };

            return in;
        }

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

};

#endif // header guard 










