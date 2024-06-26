#ifndef TBB_IIR_MULTI_CORE_H
#define TBB_IIR_MULTI_CORE_H 1

#include <array>
#include <vector>
#include <cassert>
#include <utility>
#include <memory>

// Implement IIR filter in a task-oriented system TBB that leverages multi-core processing.
template<typename V,int N> class TBBIIRMultiCore{ 

    using T = decltype(std::declval<V>().extract(0));
    static constexpr int M = V::size();
    static constexpr int L = M*M;
    using arrayV = std::array<V,M>;
    
    private:

        std::array<T,N> b1,b2,a1,a2,xi1,xi2,yi1,yi2;

    public:

        // N denotes the number of cascaded sos.
        TBBIIRMultiCore(const T (&coefs)[N][5],const T (&inits)[N][4]){

            for (int i=0;i<N;i++){

                b1[i] = coefs[i][1];
                b2[i] = coefs[i][2];
                a1[i] = coefs[i][3];
                a2[i] = coefs[i][4];
                xi1[i] = inits[i][0];
                xi2[i] = inits[i][1];
                yi1[i] = inits[i][2];
                yi2[i] = inits[i][3];
            }
        };

    inline std::pair<std::vector<T>,std::vector<T>> operator()(std::vector<T> in_data){

        // the input data to multi-core iir filter must be a multiple of M*M
        assert(in_data.size()%L == 0);

        size_t block_max = in_data.size()/L;

        // pre-process input data to transfer type from T to arrayV
        std::vector<arrayV> x;
        for (int i = 0; i < block_max; i++){
            arrayV tmp;
            for (auto n = 0; n < M; n++)
                tmp[n].load(&in_data[n*M+i*L]);
            x.push_back(tmp);
        }

        size_t n_block = 0;
        std::vector<T> output,post_inits;

        tbb::flow::graph g;

        // source node generate one data block (a matrix of samples) at one time.
        tbb::flow::source_node<DataBlock<V>> my_src(g,
        [&](DataBlock<V> &in_block)-> bool{
            if (n_block < block_max){

                in_block.data = x[n_block];
                in_block.ori_data = x[n_block];
                in_block.tag = n_block;
                n_block++;
                // attach the last flag if the last data block in input data is sent out
                if (n_block == block_max)
                    in_block.last = true;

            return true;}else{return false;}},false);

        tbb::flow::function_node<DataBlock<V>,DataBlock<V>> prior_permute(g,tbb::flow::unlimited,
        [&](DataBlock<V> v) -> DataBlock<V>{
            v.data = _permuteV(v.data);
            return v;
        });

        std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> init_adder;
        std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> zic;
        std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> rd;
        std::vector<std::unique_ptr<tbb::flow::sequencer_node<DataBlock<V>>>> seq_for_init,seq_for_buffer;
        std::vector<std::unique_ptr<Buffer<V>>> buffer_node;
        std::vector<std::unique_ptr<InterBlockRD<V>>> inter_block_rd;
        std::vector<std::unique_ptr<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>> forward;

        tbb::flow::sender<DataBlock<V>> *prev_node = &prior_permute;

        // note: the node of TBB flow graph is a very high-level construction, it is super hard to design nested function nodes for series as single core.
        for (int i=0;i<N;i++){

            seq_for_init.push_back(std::make_unique<tbb::flow::sequencer_node<DataBlock<V>>>(
                g,[](const DataBlock<V> &v) -> size_t{
                return v.tag;}));

            init_adder.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
                g,tbb::flow::serial,InitAdder<V>{xi1[i],xi2[i]}));

            zic.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
                g,tbb::flow::unlimited,NoStateZIC<V>{b1[i],b2[i],a1[i],a2[i],xi1[i],xi2[i]}));

            rd.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
                g,tbb::flow::unlimited,RecurDoubV<V>{a1[i],a2[i]}));

            seq_for_buffer.push_back(std::make_unique<tbb::flow::sequencer_node<DataBlock<V>>>(
                g,[](const DataBlock<V> &v) -> size_t{
                return v.tag;}));

            buffer_node.push_back(std::make_unique<Buffer<V>>(g,M));

            inter_block_rd.push_back(std::make_unique<InterBlockRD<V>>(g,a1[i],a2[i],yi1[i],yi2[i]));

            forward.push_back(std::make_unique<tbb::flow::function_node<DataBlock<V>,DataBlock<V>>>(
                g,tbb::flow::unlimited,ICCForward<V>{a1[i],a2[i]}));
            
            tbb::flow::make_edge(*prev_node,*seq_for_init.back());
            tbb::flow::make_edge(*seq_for_init.back(),*init_adder.back());
            tbb::flow::make_edge(*init_adder.back(),*zic.back());
            tbb::flow::make_edge(*zic.back(),*rd.back());
            tbb::flow::make_edge(*rd.back(),*seq_for_buffer.back());
            tbb::flow::make_edge(*seq_for_buffer.back(),*buffer_node.back());
            tbb::flow::make_edge(tbb::flow::output_port<0>(*buffer_node.back()),*inter_block_rd.back());
            tbb::flow::make_edge(tbb::flow::output_port<0>(*inter_block_rd.back()),*forward.back());

            prev_node = forward.back().get();

        }

        tbb::flow::function_node<DataBlock<V>,DataBlock<V>> post_permute(g,tbb::flow::serial,
        [&](DataBlock<V> v) -> DataBlock<V>{
            v.data = _permuteV(v.data);
            return v;
        });

        tbb::flow::sequencer_node<DataBlock<V>> seq_for_sink(g,
            [](const DataBlock<V> &v) -> size_t{
            return v.tag;
        });

        tbb::flow::function_node<DataBlock<V>> sink(g,tbb::flow::serial,[&](DataBlock<V> out){

            std::array<T,L> result;
            for (auto n=0; n<M; n++) out.data[n].store(&result[n*M]);
            output.insert(output.end(),result.begin(),result.end());

            if (out.last)
                post_inits = out.post_inits;

        });

        tbb::flow::make_edge(my_src,prior_permute);
        tbb::flow::make_edge(*prev_node,post_permute);
        tbb::flow::make_edge(post_permute, seq_for_sink);
        tbb::flow::make_edge(seq_for_sink, sink);

        my_src.activate();
        g.wait_for_all();

        return std::make_pair(output,post_inits);

    }

};

#endif // header guard 

