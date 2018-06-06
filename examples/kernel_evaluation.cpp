#include <evaluation.hpp>
#include <visualization.hpp>

#include "../algo/CUCCL_LE/CUCCL_LE.hpp"
#include "../algo/CUCCL_LE/CUCCL_LE.cuh"
#include "../algo/CUCCL_NP/CUCCL_NP.cuh"
#include "../algo/CUCCL_DPL/CUCCL_DPL.cuh"


#include <iostream>

#define RUNTIMETEST 1
#define CORRECTNESSTEST 1

double total_runtime_4 = 0 ;
double total_runtime_8 = 0 ; 
int    total_test_num = 0 ;
int    total_pass_num = 0 ;


std::string getOsName()
{
    #ifdef _WIN32
    return "Windows 32-bit";
    #elif _WIN64
    return "Windows 64-bit";
    #elif __unix || __unix__
    return "Unix";
    #elif __APPLE__ || __MACH__
    return "Mac OSX";
    #elif __linux__
    return "Linux";
    #elif __FreeBSD__
    return "FreeBSD";
    #else
    return "Other";
    #endif
}     


void run_test(std::string imageName, std::string cclalgo)
{

    std::cout << "  Testing CCL on image :" << current << std::endl;
    void* eval ;
    if (cclalgo == "LE")
    {
        eval = new Evaluation<CCLLEGPU>( imageName.c_str() );
    }
    else if (cclalgo == "DPL" )
    {
        eval = new Evaluation<CCLDPLGPU>( imageName.c_str());
    }
    else 
    {
        eval = new Evaluation<CCLNPGPU>( imageName.c_str() );
    }

    

    #ifdef RUNTIMETEST
        double cur_time_4 = eval->runTime(4, 0) ;
        double cur_time_8 = eval->runTime(8, 0) ;
        std::cout << "    @ Time elapsed for connectivity 4 :" << cur_time_4 << std::endl;
        std::cout << "    @ Time elapsed for connectivity 8 :" << cur_time_8 << std::endl;
        total_runtime_4 += cur_time_4 ;
        total_runtime_8 += cur_time_8 ;
    #endif

    #ifdef CORRECTNESSTEST
        std::cout << "======================= Under Construction =====================" << std::endl;
    #endif

    std::cout << std::endl;
    delete eval ;
    eval = nullptr ; 
    

    

    


}


int main( int argc, char* argv[])
{
    if (argc == 1 || argc == 2)
    {
        std::cerr << "  Invalid input, you need to provide the input image " << std::endl;
        return ;
    }


    std::string cclalgo = std::string( argv[1] ) ;

    if ( cclalgo != "DPL" && cclalgo != "NP" && cclalgo != "LE")
    {
        std::cerr << "  Invalid algorithm, you can use one of LE, NP and DPL" << std::endl;
    }


    std::cout << "  CCL algorithm validation : " << std::endl;
    std::cout << "    @ system    : " << getOsName() << std::endl; 
    std::cout << "    @ algorithm : " << cclalgo << std::endl;


    for ( int i = 2; i < argc; i++)
    {
        std::string current = std::string( argv[i] ) ;
        run_test( current.c_str(), cclalgo) ;


    }
}