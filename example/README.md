### recommand compiler flags: 
clang++ -I/usr/local/include -mavx2 -mfma -march=native -fno-trapping-math -fno-math-errno -I$VCL_PATH -I$TBB_INCLUDE -Wl,-rpath,$TBB_LIBRARY_RELEASE -L$TBB_LIBRARY_RELEASE -ltbb -std=c++20 -O3 -w -o filter filter.cpp
