#ifndef BUFFER_H
#define BUFFER_H 1

#include <vector>
#include <tbb/tbb.h>
#include "vectorclass.h"
#include "data_block.h"

// A buffer prior to inter block recursive doubling that buffer and combine data blocks in a vector.
template <typename V> class Buffer: public tbb::flow::multifunction_node<DataBlock<V>, std::tuple<std::vector<DataBlock<V>>>> {

    using InputType = DataBlock<V>; 
    using OutputType = std::vector<DataBlock<V>>; 
    using NodeType = tbb::flow::multifunction_node<InputType, std::tuple<OutputType>>;

    private:

        std::vector<DataBlock<V>> buffer;  
        tbb::flow::graph& graph; 
        size_t rd_length;

    public:

        Buffer(tbb::flow::graph& g,size_t M): NodeType(g, tbb::flow::serial, 
          [this](const InputType& item, typename NodeType::output_ports_type& op) {

                this->buffer.push_back(item);    
                // if the number of data blocks in buffer approaches a threshold (maximum SIMD length that the processor provides)
                if (this->buffer.size() == rd_length) { 

                    OutputType full_buffer(std::move(this->buffer));  
                    this->buffer.clear();  
                    std::get<0>(op).try_put(full_buffer);  
                }

                // the remaining data blocks in buffer that is below SIMD length is combined by half of the maximum SIMD length, 1/4, ...
                if (!this->buffer.empty() && item.last){

                    OutputType remain_buffer;
                    if (this->buffer.size() >= rd_length/2){
                        remain_buffer.insert(remain_buffer.end(),this->buffer.begin(),this->buffer.begin() + rd_length/2);
                        this->buffer.erase(this->buffer.begin(), this->buffer.begin() + rd_length/2);  
                        std::get<0>(op).try_put(remain_buffer);  
                        remain_buffer.clear();
                    }

                    if (this->buffer.size() >= rd_length/4 && this->buffer.size() < rd_length/2){
                        remain_buffer.insert(remain_buffer.end(),this->buffer.begin(),this->buffer.begin()+rd_length/4);
                        this->buffer.erase(this->buffer.begin(), this->buffer.begin() + rd_length/4);  
                        std::get<0>(op).try_put(remain_buffer);  
                        remain_buffer.clear();
                    }

                    if (this->buffer.size() >= rd_length/8 && this->buffer.size() < rd_length/4){
                        remain_buffer.insert(remain_buffer.end(),this->buffer.begin(),this->buffer.begin()+rd_length/8);
                        this->buffer.erase(this->buffer.begin(), this->buffer.begin() + rd_length/8);  
                        std::get<0>(op).try_put(remain_buffer);  
                        remain_buffer.clear();
                    } 
                }
            }
        ), graph(g),rd_length(M) {}        
};



#endif // BUFFER_H


