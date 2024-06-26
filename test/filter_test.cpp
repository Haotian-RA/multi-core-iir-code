#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include <tbb/tbb.h>
#include "recursive_filter.h"
#include <numeric>
#include <memory>
#include <iterator>

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

    constexpr size_t n_in = 100, N = 3;
    constexpr size_t len = n_in*L+2*M+1; // 100*64+2*8+1
    std::vector<T> data(len);
    std::iota(data.begin(), data.end(), 0);

    std::array<T,len> ex_result;
    std::vector<T> result(len);

    IirCoreOrderTwo<V> IIR1(b1, b2, a1, a2, xi1, xi2, yi1, yi2), IIR2(b1, b2, a1, a2, xi1, xi2, yi1, yi2), IIR3(b1, b2, a1, a2, xi1, xi2, yi1, yi2);

    for (auto n = 0; n < len; n++)
        ex_result[n] = IIR3.benchmark(IIR2.benchmark(IIR1.benchmark(data[n])));

    // define array of coefficients and initial conditions
    T coefs[N][5] = {1,b1,b2,a1,a2,1,b1,b2,a1,a2,1,b1,b2,a1,a2}; 
    T inits[N][4] = {xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2,xi1,xi2,yi1,yi2};

    auto multi_core_filter = makeMultiCoreFilter(coefs,inits);
    multi_core_filter(data.begin(),data.end(),result.begin());

    for (int i=0;i<len;i++) 
        CHECK(result[i] == doctest::Approx(ex_result[i]));    

};

TEST_SUITE_END();

#endif // doctest







