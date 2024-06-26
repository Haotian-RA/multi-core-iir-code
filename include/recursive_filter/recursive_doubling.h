#ifndef RECURSIVE_DOUBLING_H
#define RECURSIVE_DOUBLING_H 1

#include "vectorclass.h"
#include "data_block.h"

// stateless vector recursive doubling 
template<typename V> class RecurDoubV{

    using T = decltype(std::declval<V>().extract(0));
    constexpr static int M = V::size();

    private: 

        T _a1, _a2;

        // vectors contain the elements at the four positions of C, C^2, C^3 ...
        V _h_22, _h_12, _h_21, _h_11;

        // pre-compute the vectors including C for RD first recursion
        V _rd1_22, _rd1_12, _rd1_21, _rd1_11; 

        // pre-compute the vectors including C for RD second recursion
        V _rd2_22, _rd2_12, _rd2_21, _rd2_11;

        // pre-compute the vectors including C for RD third recursion
        V _rd3_22, _rd3_12, _rd3_21, _rd3_11;

        // vectors in matrix A, A=[h2 h1].
        V _h2, _h1;

    public:

        RecurDoubV(const T a1, const T a2): _a1(a1), _a2(a2) { 

            // pre-compute matrix A.
            impulse_response();

            // pre-compute the vectors including C in recursive doubling.
            rd_vectors();

        };

    inline DataBlock<V> operator()(DataBlock<V> in){
        
        std::array<V,2> v;                

        // step 2: first recursion
        V b2 = permute8<-1,0,-1,2,-1,4,-1,6>(in.data[M-2]);
        V b1 = permute8<-1,0,-1,2,-1,4,-1,6>(in.data[M-1]);

        v[0] = mul_add(b2, _rd1_22, in.data[M-2]);
        v[0] = mul_add(b1, _rd1_12, v[0]);
        v[1] = mul_add(b2, _rd1_21, in.data[M-1]);
        v[1] = mul_add(b1, _rd1_11, v[1]);

        // step 3: second recursion
        b2 = permute8<-1,-1,1,1,-1,-1,5,5>(v[0]);
        b1 = permute8<-1,-1,1,1,-1,-1,5,5>(v[1]);

        v[0] = mul_add(b2, _rd2_22, v[0]);
        v[0] = mul_add(b1, _rd2_12, v[0]);
        v[1] = mul_add(b2, _rd2_21, v[1]);
        v[1] = mul_add(b1, _rd2_11, v[1]);

        // step 4: third recursion
        b2 = permute8<-1,-1,-1,-1,3,3,3,3>(v[0]);
        b1 = permute8<-1,-1,-1,-1,3,3,3,3>(v[1]);

        v[0] = mul_add(b2, _rd3_22, v[0]);
        v[0] = mul_add(b1, _rd3_12, v[0]);
        v[1] = mul_add(b2, _rd3_21, v[1]);
        v[1] = mul_add(b1, _rd3_11, v[1]);

        in.data[M-2] = *&v[0];
        in.data[M-1] = *&v[1];

        return in;
        
    };

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
    inline void rd_vectors() {

        C_power();

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

#endif // header guard 










