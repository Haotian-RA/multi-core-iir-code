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

const T b1 = 0.1, b2 = -0.5, a1 = 0.2, a2 = 0.3, xi1 = 2, xi2 = 3, yi1 = -0.5, yi2 = 1.5;

TEST_CASE("serial concurrency:"){

    constexpr size_t n_in = 8, N = 2;

    // use duplicate sequence to test whether state is maintained.
    std::array<T,n_in*L> data;
    for (int i=0;i<n_in;i++)
        std::iota(data.begin()+i*L, data.begin()+(i+1)*L, 0);

    std::array<arrayV,n_in> x;
    for (int i=0;i<n_in;i++) {
        for (auto n=0;n<M;n++) x[i][n].load(&data[n*M+i*M*M]);

    }

    // for examining
    IirCoreOrderTwo<V> IIR1(b1,b2,a1,a2,xi1,xi2,yi1,yi2),IIR2(b1,b2,a1,a2,xi1,xi2,yi1,yi2);

    std::array<T,n_in*L> ex_result; 
    std::array<T,L> result;

    for (auto n=0; n<n_in*M*M; n++) ex_result[n] = IIR2.benchmark(IIR1.benchmark(data[n]));

    size_t block_max = n_in;
    size_t n_block = 0;
    size_t count = -1;

    tbb::flow::graph g;

    tbb::flow::source_node<arrayV> my_src(g,
    [&](arrayV &in)-> bool{
        if (n_block < block_max){

            in = x[n_block];
            n_block++;

        return true;}else{return false;}},false);

    tbb::flow::function_node<arrayV,arrayV> prior_permute(g,tbb::flow::serial,_permuteV<V>);

    std::array<std::shared_ptr<ZeroInitCond<V>>, N> ZIC_array;
    std::array<std::shared_ptr<InitCondCorc<V>>, N> ICC_array;

    for (size_t i = 0; i < N; ++i) {
        ZIC_array[i] = std::make_shared<ZeroInitCond<V>>(b1, b2, a1, a2, xi1, xi2);
        ICC_array[i] = std::make_shared<InitCondCorc<V>>(a1, a2, yi1, yi2);
    }

    size_t num = -1;

    // A vector of unique pointers to for varied (cascaded nodes)
    std::vector<std::shared_ptr<tbb::flow::function_node<arrayV, arrayV>>> series(N);
    for (size_t i = 0; i < N; ++i) {
        series[i] = std::make_shared<tbb::flow::function_node<arrayV, arrayV>>(g, tbb::flow::serial, 
        [&, i](arrayV v) -> arrayV {
            return ICC_array[i]->ICC_T(ZIC_array[i]->ZIC_T(v));
        });
    }

    tbb::flow::function_node<arrayV,arrayV> post_permute(g,tbb::flow::serial,_permuteV<V>);

    tbb::flow::function_node<arrayV> sink(g,tbb::flow::serial,[&](arrayV out){

        count++;

        for (auto n=0; n<M; n++) out[n].store(&result[n*M]);

        for (auto n=0; n<L; n++) CHECK(result[n] == doctest::Approx(ex_result[n+count*M*M]));

    });


    tbb::flow::make_edge(my_src,prior_permute);
    tbb::flow::make_edge(prior_permute,*series[0]);

    for (size_t i = 0; i < N - 1; ++i) {
        tbb::flow::make_edge(*series[i],*series[i+1]);
    }

    tbb::flow::make_edge(*series[N-1],post_permute);
    tbb::flow::make_edge(post_permute,sink); 

    my_src.activate();
    g.wait_for_all();

};

TEST_SUITE_END();

#endif // doctest
