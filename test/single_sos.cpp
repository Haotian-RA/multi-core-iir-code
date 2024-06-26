#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <tbb/tbb.h>
#include "doctest.h"
#include "recursive_filter.h"
#include <numeric>


#ifdef DOCTEST_LIBRARY_INCLUDED

#define TBB_DEPRECATED_LIMITER_NODE_CONSTRUCTOR 1
#define TBB_PREVIEW_FLOW_GRAPH_NODES 1

TEST_CASE("TBB implementation test for single sos:"){

    using V = Vec8f;
    using T = decltype(std::declval<V>().extract(0));

    constexpr int M = V::size();
    constexpr int L = M*M;

    using arrayV = std::array<V,M>;
    using namespace std::placeholders;

    const float b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3, xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

    std::array<T,L> data;
    std::iota(data.begin(), data.end(), 0);

    arrayV x;
    for (auto n=0; n<M; n++) x[n].load(&data[n*M]);

    // for examining
    IirCoreOrderTwo<V> IIR(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T,L> ex_result, result;
    for (auto n=0; n<M*M; n++) ex_result[n] = IIR.benchmark(data[n]);

    ZeroInitCond<V> ZIC(b1,b2,a1,a2,xi1,xi2);
    InitCondCorc<V> ICC(a1,a2,yi1,yi2);

    size_t block_max = 1;
    size_t n_block = 0;

    tbb::flow::graph g;

    tbb::flow::source_node<arrayV> my_src(g,
    [&](arrayV &in)-> bool{
        if (n_block < block_max){

            in = x;
            n_block++;

        return true;}else{return false;}},false);

    tbb::flow::function_node<arrayV,arrayV> prior_permute(g,tbb::flow::serial,_permuteV<V>);

    tbb::flow::function_node<arrayV,arrayV> zic(g,tbb::flow::serial,
        [&ZIC](arrayV v) -> arrayV {
            return ZIC.ZIC_T(v);
        }
    );

    tbb::flow::function_node<arrayV,arrayV> icc(g,tbb::flow::serial,
        [&ICC](arrayV v) -> arrayV {
            return ICC.ICC_T(v);
        }
    );

    tbb::flow::function_node<arrayV,arrayV> post_permute(g,tbb::flow::serial,_permuteV<V>);

    tbb::flow::function_node<arrayV> sink(g,tbb::flow::serial,[&](arrayV out){

        for (auto n=0; n<M; n++) out[n].store(&result[n*M]);

        for (auto n=0; n<L; n++) CHECK(result[n] == doctest::Approx(ex_result[n]));

    });


    tbb::flow::make_edge(my_src,prior_permute);
    tbb::flow::make_edge(prior_permute,zic);
    tbb::flow::make_edge(zic,icc);
    tbb::flow::make_edge(icc,post_permute);
    tbb::flow::make_edge(post_permute,sink); 

    my_src.activate();
    g.wait_for_all();

};

TEST_SUITE_END();

#endif // doctest
