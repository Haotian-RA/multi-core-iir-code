#ifndef INTER_BLOCK_RD_H
#define INTER_BLOCK_RD_H 1

#include "vectorclass.h"
#include "data_block.h"
#include "shift_reg.h"
#include <cmath>
#include <vector>

// Compute initials for multiple (length of SIMD) blocks by recursive doubling.
template <typename V> class InterBlockRD: public tbb::flow::multifunction_node<std::vector<DataBlock<V>>, std::tuple<DataBlock<V>>> {

    using T = decltype(std::declval<V>().extract(0));
    constexpr static int M = V::size();

    private:

        T _a1, _a2;

        Shift<V> _S;

        // vectors in matrix A, A=[h2 h1].
        V _h2, _h1;

        // pre-compute the vectors including C for recursive doubling initialization for M = 8
        V _rd0_22, _rd0_12, _rd0_21, _rd0_11; 

        // pre-compute the vectors including C for RD first recursion
        V _rd1_22, _rd1_12, _rd1_21, _rd1_11; 

        // pre-compute the vectors including C for RD second recursion
        V _rd2_22, _rd2_12, _rd2_21, _rd2_11;

        // pre-compute the vectors including C for RD third recursion
        V _rd3_22, _rd3_12, _rd3_21, _rd3_11;

        // pre-compute the vectors including C for recursive doubling initialization for M = 4
        Vec4f _rd40_22, _rd40_12, _rd40_21, _rd40_11; 

        // pre-compute the vectors including C for RD first recursion
        Vec4f _rd41_22, _rd41_12, _rd41_21, _rd41_11; 

        // pre-compute the vectors including C for RD second recursion
        Vec4f _rd42_22, _rd42_12, _rd42_21, _rd42_11;
 
        // static const size_t C_len = M*std::pow(2,std::log2(M)-1);
        static const size_t C_len = M * (M >> 1);

        std::array<T,C_len> h_22, h_12, h_21, h_11;

    public: 

        InterBlockRD(tbb::flow::graph& g,T a1, T a2, T yi1, T yi2)
        :tbb::flow::multifunction_node<std::vector<DataBlock<V>>, std::tuple<DataBlock<V>>>(
            g, tbb::flow::serial,
            [this](const std::vector<DataBlock<V>>& in, typename InterBlockRD::output_ports_type& ports){

                // attach initial conditions for multiple blocks based on the size of incoming data
                if (in.size() == M){

                    V yi2(in[0].data[M-2][M-1],in[1].data[M-2][M-1],
                        in[2].data[M-2][M-1],in[3].data[M-2][M-1],
                        in[4].data[M-2][M-1],in[5].data[M-2][M-1],
                        in[6].data[M-2][M-1],in[7].data[M-2][M-1]); 
                    V yi1(in[0].data[M-1][M-1],in[1].data[M-1][M-1],
                        in[2].data[M-1][M-1],in[3].data[M-1][M-1],
                        in[4].data[M-1][M-1],in[5].data[M-1][M-1],
                        in[6].data[M-1][M-1],in[7].data[M-1][M-1]);

                    // recursive doubling step 1: initialization
                    yi2 = mul_add(_rd0_22, _S[-2], yi2);
                    yi2 = mul_add(_rd0_12, _S[-1], yi2);
                    yi1 = mul_add(_rd0_21, _S[-2], yi1);
                    yi1 = mul_add(_rd0_11, _S[-1], yi1);

                    // step 2: first recursion
                    V b2 = permute8<-1,0,-1,2,-1,4,-1,6>(yi2);
                    V b1 = permute8<-1,0,-1,2,-1,4,-1,6>(yi1);

                    yi2 = mul_add(b2, _rd1_22, yi2);
                    yi2 = mul_add(b1, _rd1_12, yi2);
                    yi1 = mul_add(b2, _rd1_21, yi1);
                    yi1 = mul_add(b1, _rd1_11, yi1);

                    // step 3: second recursion
                    b2 = permute8<-1,-1,1,1,-1,-1,5,5>(yi2);
                    b1 = permute8<-1,-1,1,1,-1,-1,5,5>(yi1);

                    yi2 = mul_add(b2, _rd2_22, yi2);
                    yi2 = mul_add(b1, _rd2_12, yi2);
                    yi1 = mul_add(b2, _rd2_21, yi1);
                    yi1 = mul_add(b1, _rd2_11, yi1);

                    // step 4: third recursion
                    b2 = permute8<-1,-1,-1,-1,3,3,3,3>(yi2);
                    b1 = permute8<-1,-1,-1,-1,3,3,3,3>(yi1);

                    yi2 = mul_add(b2, _rd3_22, yi2);
                    yi2 = mul_add(b1, _rd3_12, yi2);
                    yi1 = mul_add(b2, _rd3_21, yi1);
                    yi1 = mul_add(b1, _rd3_11, yi1);

                    V y_inits2 = blend8<8,0,1,2,3,4,5,6>(yi2, _S[-2]);
                    V y_inits1 = blend8<8,0,1,2,3,4,5,6>(yi1, _S[-1]);

                    // shift the last two ys for next array of blocks
                    _S.shift(yi2[M-1]);
                    _S.shift(yi1[M-1]);

                    for (DataBlock<V> data: in) {

                        data.y_inits[0] = y_inits2[data.tag % M];
                        data.y_inits[1] = y_inits1[data.tag % M];

                        if (data.last){

                            data.post_inits.push_back(_S[-2]);
                            data.post_inits.push_back(_S[-1]);
                        };

                        std::get<0>(ports).try_put(data);  // Emit each modified element
                    }
                };

                if (in.size() == M/2){

                    Vec4f yi2(in[0].data[M-2][M-1],in[1].data[M-2][M-1],
                        in[2].data[M-2][M-1],in[3].data[M-2][M-1]); 
                    Vec4f yi1(in[0].data[M-1][M-1],in[1].data[M-1][M-1],
                        in[2].data[M-1][M-1],in[3].data[M-1][M-1]);

                    // recursive doubling step 1: initialization
                    yi2 = mul_add(_rd40_22, _S[-2], yi2);
                    yi2 = mul_add(_rd40_12, _S[-1], yi2);
                    yi1 = mul_add(_rd40_21, _S[-2], yi1);
                    yi1 = mul_add(_rd40_11, _S[-1], yi1);

                    // step 2: first recursion
                    Vec4f b2 = permute4<-1,0,-1,2>(yi2);
                    Vec4f b1 = permute4<-1,0,-1,2>(yi1);

                    yi2 = mul_add(b2, _rd41_22, yi2);
                    yi2 = mul_add(b1, _rd41_12, yi2);
                    yi1 = mul_add(b2, _rd41_21, yi1);
                    yi1 = mul_add(b1, _rd41_11, yi1);

                    // step 3: second recursion
                    b2 = permute4<-1,-1,1,1>(yi2);
                    b1 = permute4<-1,-1,1,1>(yi1);

                    yi2 = mul_add(b2, _rd42_22, yi2);
                    yi2 = mul_add(b1, _rd42_12, yi2);
                    yi1 = mul_add(b2, _rd42_21, yi1);
                    yi1 = mul_add(b1, _rd42_11, yi1);

                    Vec4f y_inits2 = blend4<4,0,1,2>(yi2, _S[-2]);
                    Vec4f y_inits1 = blend4<4,0,1,2>(yi1, _S[-1]);

                    // shift the last two ys for next array of blocks
                    _S.shift(yi2[M/2-1]);
                    _S.shift(yi1[M/2-1]);

                    for (DataBlock<V> data: in) {

                        data.y_inits[0] = y_inits2[data.tag % (M/2)];
                        data.y_inits[1] = y_inits1[data.tag % (M/2)];

                        if (data.last){

                            data.post_inits.push_back(_S[-2]);
                            data.post_inits.push_back(_S[-1]);
                        };

                        std::get<0>(ports).try_put(data);  // Emit each modified element
                    }
                };


                if (in.size() == M/4){

                    T vi2 = h_22[M-1]*_S[-2]+h_12[M-1]*_S[-1]+in[0].data[M-2][M-1];
                    T vi1 = h_21[M-1]*_S[-2]+h_11[M-1]*_S[-1]+in[0].data[M-1][M-1];

                    T yi2 = h_22[M-1]*vi2+h_12[M-1]*vi1+in[1].data[M-2][M-1];
                    T yi1 = h_21[M-1]*vi2+h_11[M-1]*vi1+in[1].data[M-1][M-1];

                    std::array<T,M/4> y_inits2,y_inits1;

                    y_inits2[0] = _S[-2];
                    y_inits2[1] = vi2;
                    y_inits1[0] = _S[-1];
                    y_inits1[1] = vi1;

                    // shift the last two ys for next array of blocks
                    _S.shift(yi2);
                    _S.shift(yi1);

                    for (DataBlock<V> data: in) {

                        data.y_inits[0] = y_inits2[data.tag % (M/4)];
                        data.y_inits[1] = y_inits1[data.tag % (M/4)];

                        if (data.last){

                            data.post_inits.push_back(_S[-2]);
                            data.post_inits.push_back(_S[-1]);
                        };

                        std::get<0>(ports).try_put(data);  // Emit each modified element
                    }
                };

                if (in.size() == M/8){

                    T yi2 = h_22[M-1]*_S[-2]+h_12[M-1]*_S[-1]+in[0].data[M-2][M-1];
                    T yi1 = h_21[M-1]*_S[-2]+h_11[M-1]*_S[-1]+in[0].data[M-1][M-1];

                    T y_inits2 = _S[-2];
                    T y_inits1 = _S[-1];

                    // shift the last two ys for next array of blocks
                    _S.shift(yi2);
                    _S.shift(yi1);

                    for (DataBlock<V> data: in) {

                        data.y_inits[0] = y_inits2;
                        data.y_inits[1] = y_inits1;

                        if (data.last){

                            data.post_inits.push_back(_S[-2]);
                            data.post_inits.push_back(_S[-1]);
                        };

                        std::get<0>(ports).try_put(data);  // Emit each modified element
                    }
                };
 
            }),
            _a1(a1),_a2(a2) {

                _S.shift(yi2);
                _S.shift(yi1);

                // pre-compute matrix A.
                impulse_response();

                // pre-compute the vectors including C in recursive doubling.
                recursive_doubling_vectors();
                
            }

        
        // use old method first try to make everything work.

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
        
            h_22[0] = _h2[M-2];
            h_12[0] = _h1[M-2];
            h_21[0] = _h2[M-1];
            h_11[0] = _h1[M-1];

            for (auto n=1; n<C_len; n++) {
                h_22[n] = _h2[M-2]*h_22[n-1] + _h2[M-1]*h_12[n-1];
                h_12[n] = _h1[M-2]*h_22[n-1] + _h1[M-1]*h_12[n-1];
                h_21[n] = _h2[M-2]*h_21[n-1] + _h2[M-1]*h_11[n-1];
                h_11[n] = _h1[M-2]*h_21[n-1] + _h1[M-1]*h_11[n-1];
            }
        };

        // calculate the vectors including elements of C in recursive doubling
        inline void recursive_doubling_vectors() {

            C_power(); // [C C^2 C^3 ... C^8 C^16 C^32]

            // RD initialization, [C^8 0 0 0 0 0 0 0] for vector size = 8
            _rd0_22 = V(h_22[M-1],0,0,0,0,0,0,0);
            _rd0_12 = V(h_12[M-1],0,0,0,0,0,0,0);
            _rd0_21 = V(h_21[M-1],0,0,0,0,0,0,0);
            _rd0_11 = V(h_11[M-1],0,0,0,0,0,0,0);

            // RD recursion 1, [0 C^8 0 C^8 0 C^8 0 C^8]
            _rd1_22 = V(0,h_22[M-1],0,h_22[M-1],0,h_22[M-1],0,h_22[M-1]);
            _rd1_12 = V(0,h_12[M-1],0,h_12[M-1],0,h_12[M-1],0,h_12[M-1]);
            _rd1_21 = V(0,h_21[M-1],0,h_21[M-1],0,h_21[M-1],0,h_21[M-1]);
            _rd1_11 = V(0,h_11[M-1],0,h_11[M-1],0,h_11[M-1],0,h_11[M-1]);

            // RD recursion 2, [0 0 C^8 C^16 0 0 C^8 C^16]
            _rd2_22 = V(0,0,h_22[M-1],h_22[2*M-1],0,0,h_22[M-1],h_22[2*M-1]);
            _rd2_12 = V(0,0,h_12[M-1],h_12[2*M-1],0,0,h_12[M-1],h_12[2*M-1]);
            _rd2_21 = V(0,0,h_21[M-1],h_21[2*M-1],0,0,h_21[M-1],h_21[2*M-1]);
            _rd2_11 = V(0,0,h_11[M-1],h_11[2*M-1],0,0,h_11[M-1],h_11[2*M-1]);

            // RD recursion 3, [0 0 0 0 C^8 C^16 C^24 C^32]
            _rd3_22 = V(0,0,0,0,h_22[M-1],h_22[2*M-1],h_22[3*M-1],h_22[4*M-1]);
            _rd3_12 = V(0,0,0,0,h_12[M-1],h_12[2*M-1],h_12[3*M-1],h_12[4*M-1]);
            _rd3_21 = V(0,0,0,0,h_21[M-1],h_21[2*M-1],h_21[3*M-1],h_21[4*M-1]);
            _rd3_11 = V(0,0,0,0,h_11[M-1],h_11[2*M-1],h_11[3*M-1],h_11[4*M-1]);

            // RD initialization, [C^8 0 0 0 0 0 0 0] for vector size = 4
            _rd40_22 = Vec4f(h_22[M-1],0,0,0);
            _rd40_12 = Vec4f(h_12[M-1],0,0,0);
            _rd40_21 = Vec4f(h_21[M-1],0,0,0);
            _rd40_11 = Vec4f(h_11[M-1],0,0,0);

            // RD recursion 1, [0 C^8 0 C^8 0 C^8 0 C^8]
            _rd41_22 = Vec4f(0,h_22[M-1],0,h_22[M-1]);
            _rd41_12 = Vec4f(0,h_12[M-1],0,h_12[M-1]);
            _rd41_21 = Vec4f(0,h_21[M-1],0,h_21[M-1]);
            _rd41_11 = Vec4f(0,h_11[M-1],0,h_11[M-1]);

            // RD recursion 2, [0 0 C^8 C^16 0 0 C^8 C^16]
            _rd42_22 = Vec4f(0,0,h_22[M-1],h_22[2*M-1]);
            _rd42_12 = Vec4f(0,0,h_12[M-1],h_12[2*M-1]);
            _rd42_21 = Vec4f(0,0,h_21[M-1],h_21[2*M-1]);
            _rd42_11 = Vec4f(0,0,h_11[M-1],h_11[2*M-1]);
       
        };

};

#endif // header guard 










