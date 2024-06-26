#ifndef MULTI_CORE_FILTER_H
#define MULTI_CORE_FILTER_H 1

#include "tbb_iir_multi_core.h"
#include <vector>
#include <tuple>

// real function to user: use the cascaded second order filter to process a trunk of data.
template<typename T,int N> class MultiCoreFilter{ 
    
    // select the vector length and type based on the requested instruction set and the type T
    #if INSTRSET >= 9  // AVX512
        using V = typename std::conditional<std::is_same<T, float>::value, Vec16f, Vec8d>::type;
    #elif INSTRSET >= 7  // AVX2
        using V = typename std::conditional<std::is_same<T, float>::value, Vec8f, Vec4d>::type;
    #else // SSE
        using V = typename std::conditional<std::is_same<T, float>::value, Vec4f, Vec2d>::type;
    #endif

    constexpr static int M = V::size();

    private:

        // multi-core filter
        TBBIIRMultiCore<V, N> _MC;

        // single-core filter
        using Series_t = decltype(series_from_coeffs<T,V>(std::declval<const T (&)[N][5]>(), std::declval<const T (&)[N][4]>())); 
        Series_t _S;

    public:

        MultiCoreFilter(const T (&coefs)[N][5],const T (&inits)[N][4]): _MC{coefs,inits},_S(series_from_coeffs<T,V>(coefs, inits)){}
    
    template<typename InputIt,typename OutputIt> inline OutputIt operator()(InputIt first,InputIt last,OutputIt d_first){

        // if the number of input samples is the multiple of M^2, then go multi-core multi-block filtering.
        if (first <= last - M*M){

            std::vector<T> input;
            std::pair<std::vector<T>,std::vector<T>> output;

            auto d = std::distance(first,last)/(M*M)*(M*M);
            input.insert(input.begin(),first,first+d);
            output = _MC(input); // std::pair(results,post_inits)

            std::copy(output.first.begin(), output.first.end(), d_first);

            T inits[N][4];
            for (int i=0;i<N;i++)
                for (int m=0;m<4;m++)
                    inits[i][m] = output.second[4*i+m];
                
            auto& series = _S.get_series_state();

            // Apply the inits_refresh function to each element in the tuple
            for_each_in_tuple(series, [&](auto& sos) {
                static int i = 0; // Static to retain value between calls
                sos.inits_refresh(inits[i]);
                ++i;
            });

            first += d;
            d_first += d;

        }

        // if the number of input samples is less than M^2 but greater than M.
        V x, y;
        while (first <= last - M){

            x.load(&*first);  
            
            y = _S.series_option1(x);

            y.store(&*d_first);

            first += M;
            d_first += M;

        }

        // if the number of input samples is less than M then do scalar operation.
        while (first <= last - 1){
            
            *d_first = _S.series_scalar(*first);

            first += 1;
            d_first += 1;

        }

        return d_first;
    }
};


// Factory function to create MultiCoreFilter instances
template<typename T, int N> MultiCoreFilter<T, N> makeMultiCoreFilter(const T (&coefs)[N][5], const T (&inits)[N][4]) {
    return MultiCoreFilter<T, N>(coefs, inits);
}

#endif // header guard 
