#include <evaluation.hpp>
#include <visualization.hpp>

#include "../algo/CUCCL_LE/CUCCL_LE.hpp"
#include "../algo/CUCCL_LE/CUCCL_LE.cuh"
#include "../algo/CUCCL_NP/CUCCL_NP.cuh"
#include "../algo/CUCCL_DPL/CUCCL_DPL.cuh"


#include <iostream>

#define RUNTIMETEST 
#define CORRECTNESSTEST


void run_test()
{
    

}


int main( int argc, char*[] argv)
{
    if (argc == 1)
    {
        std::cerr << "  Invalid input, you need to provide the input image " << std::endl;
        return ;
    }

    double total_runtime = 0 ; 
    int    total_test_num = 0 ;
    int    total_pass_num = 0 ;
    

    
    

}