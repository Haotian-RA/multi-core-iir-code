#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include <tbb/tbb.h>
#include "recursive_filter.h"
#include <numeric>
#include <memory>

#ifdef DOCTEST_LIBRARY_INCLUDED

#define TBB_DEPRECATED_LIMITER_NODE_CONSTRUCTOR 1
#define TBB_PREVIEW_FLOW_GRAPH_NODES 1

TEST_SUITE_BEGIN("TBB implementation test for cascaded sos:");

using V = Vec8f;
using T = decltype(std::declval<V>().extract(0));

constexpr int M = V::size();
constexpr int L = M*M;

using arrayV = std::array<V,M>;
using namespace std::placeholders;

const T b1 = 0.1, b2 = -0.5, a1 = 0.5, a2 = 0.3, xi1 = 1, xi2 = 4, yi1 = -0.2, yi2 = 2.5;

TEST_CASE("unlimited concurrency:"){

    constexpr size_t n_in = 100, N = 2;
    std::array<T, n_in * L> data;
    std::iota(data.begin(), data.end(), 0);

    std::array<arrayV, n_in> x;
    for (int i = 0; i < n_in; i++)
        for (auto n = 0; n < M; n++)
            x[i][n].load(&data[n * M + i * M * M]);

    IirCoreOrderTwo<V> IIR1(b1, b2, a1, a2, xi1, xi2, yi1, yi2), IIR2(b1, b2, a1, a2, xi1, xi2, yi1, yi2);
    std::array<T, n_in * L> ex_result;
    std::array<T, L> result;

    for (auto n = 0; n < n_in * M * M; n++)
        ex_result[n] = IIR2.benchmark(IIR1.benchmark(data[n]));

    /* 
        simple design principle:
        1. use a simple buffer to store inits of inputs.
        2. send an object of data that always include the inits.
        3. modify zic from stateful to stateless.

     */

    size_t block_max = n_in;
    size_t n_block = 0;
    size_t count = -1;

    tbb::flow::graph g;

    tbb::flow::source_node<DataBlock<V>> my_src(g,
    [&](DataBlock<V> &in)-> bool{
        if (n_block < block_max){

            in.data = x[n_block];
            in.tag = n_block;
            n_block++;
            if (n_block == block_max)
                in.last = true;

        return true;}else{return false;}},false);

    tbb::flow::function_node<DataBlock<V>,DataBlock<V>> prior_permute(g,tbb::flow::unlimited,
        [&](DataBlock<V> v) -> DataBlock<V>{
            v.data = _permuteV(v.data);
            return v;
        }
    
    );

    std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> init_adder;
    std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> zic;
    std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> rd;
    std::vector<std::unique_ptr<tbb::flow::sequencer_node<DataBlock<V>>>> prior_sequencer,sequencer;
    std::vector<std::unique_ptr<Buffer<V>>> buffer_node;
    std::vector<std::unique_ptr<InterBlockRD<V>>> inter_block_rd;
    std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> forward;

    tbb::flow::sender<DataBlock<V>> *prev_node = &prior_permute;

    for (int i=0;i<N;i++){

        prior_sequencer.push_back(std::make_unique<tbb::flow::sequencer_node<DataBlock<V>>>(
            g,[](const DataBlock<V> &v) -> size_t{
            return v.tag;
        }));

        init_adder.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
            g,tbb::flow::serial,InitAdder<V>{xi1,xi2}));

        zic.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
            g,tbb::flow::unlimited,NoStateZIC<V>{b1,b2,a1,a2,xi1,xi2}));

        rd.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
            g,tbb::flow::unlimited,RecurDoubV<V>{a1,a2}));

        sequencer.push_back(std::make_unique<tbb::flow::sequencer_node<DataBlock<V>>>(
            g,[](const DataBlock<V> &v) -> size_t{
            return v.tag;
        }));

        buffer_node.push_back(std::make_unique<Buffer<V>>(g,M));

        inter_block_rd.push_back(std::make_unique<InterBlockRD<V>>(g,a1,a2,yi1,yi2));

        forward.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
            g,tbb::flow::unlimited,ICCForward<V>{a1,a2}));
        
        tbb::flow::make_edge(*prev_node,*prior_sequencer.back());
        tbb::flow::make_edge(*prior_sequencer.back(),*init_adder.back());
        tbb::flow::make_edge(*init_adder.back(),*zic.back());
        tbb::flow::make_edge(*zic.back(),*rd.back());
        tbb::flow::make_edge(*rd.back(),*sequencer.back());
        tbb::flow::make_edge(*sequencer.back(),*buffer_node.back());
        tbb::flow::make_edge(tbb::flow::output_port<0>(*buffer_node.back()),*inter_block_rd.back());
        tbb::flow::make_edge(tbb::flow::output_port<0>(*inter_block_rd.back()),*forward.back());

        prev_node = forward.back().get();


    }

    tbb::flow::function_node<DataBlock<V>,DataBlock<V>> post_permute(g,tbb::flow::serial,
        [&](DataBlock<V> v) -> DataBlock<V>{
            v.data = _permuteV(v.data);
            return v;
        }
    
    );


    tbb::flow::sequencer_node<DataBlock<V>> sequencer2(g,
        [](const DataBlock<V> &v) -> size_t{
        return v.tag;
    });

    tbb::flow::function_node<DataBlock<V>> sink(g,tbb::flow::serial,[&](DataBlock<V> out){

        for (auto n=0; n<M; n++) out.data[n].store(&result[n*M]);

        for (auto n=0; n<L; n++){

            CHECK(result[n] == doctest::Approx(ex_result[n+out.tag*L]));
            
        }
    });

    tbb::flow::make_edge(my_src,prior_permute);

    tbb::flow::make_edge(*prev_node,post_permute);

    tbb::flow::make_edge(post_permute, sequencer2);

    tbb::flow::make_edge(sequencer2, sink);

    my_src.activate();
    g.wait_for_all();

};

TEST_SUITE_END();

#endif // doctest
