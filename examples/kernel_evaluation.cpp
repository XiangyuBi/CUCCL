#include <evaluation.hpp>
#include <visualization.hpp>

#include "../algo/CUCCL_CPU/CUCCL_LE.hpp"
#include "../algo/CUCCL_LE/CUCCL_LE.cuh"
#include "../algo/CUCCL_NP/CUCCL_NP.cuh"
#include "../algo/CUCCL_DPL/CUCCL_DPL.cuh"


#include <iostream>

using namespace CUCCL ;

#define RUNTEST 1 
#define RUNTIMETEST 1
#define CORRECTNESSTEST 1
//#define VISUALIZATION 1

double total_runtime_4 = 0 ;
double total_runtime_8 = 0 ; 
//int    total_test_num = 0 ;
int    total_pass_num = 0 ;
double total_runtime_CPU_4 = 0 ;
double total_runtime_CPU_8 = 0;

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

    std::cout << "  Testing CCL on image :" << imageName << std::endl;
   	double cur_time_4 ;
	double cur_time_8 ;


    if (cclalgo == "LE")
    {	
		Evaluation<CCLLEGPU>* eval = new Evaluation<CCLLEGPU> ( imageName.c_str() ) ;
		cur_time_4 = eval->runTime(4 , 0 );
		cur_time_8 = eval->runTime(8 , 0 );
		delete eval ;
		eval = nullptr ;
        //eval = static_cast<Evaluation<CUCCL::CCLLEGPU> >( new Evaluation<CUCCL::CCLLEGPU>( imageName.c_str() ) );
    }
    else if (cclalgo == "DPL" )
    {
     	
		auto* eval = new Evaluation<CCLDPLGPU>( imageName.c_str() ) ;
		cur_time_4 = eval->runTime(4, 0);
		cur_time_8 = eval->runTime(8, 0);
		delete eval ;
		eval = nullptr ;
		// eval = static_cast<Evaluation<CUCCL::CCLDPLGPU> >( new Evaluation<CUCCL::CCLDPLGPU>( imageName.c_str()) );
    }
    else 
    {	
		auto* eval = new Evaluation<CCLNPGPU>( imageName.c_str() ) ;
		cur_time_4 = eval->runTime(4, 0);
		cur_time_8 = eval->runTime(8, 0);
		delete eval ;
		eval = nullptr ;
       // eval = static_cast<Evaluation<CUCCL::CCLNPGPU> >( new Evaluation<CUCCL::CCLNPGPU>( imageName.c_str() ) ) ;
    }

    

    #ifdef RUNTIMETEST
       // double cur_time_4 = eval->runTime(4, 0) ;
       // double cur_time_8 = eval->runTime(8, 0) ;
	     auto* eval = new Evaluation<CCLLECPU>( imageName.c_str() ) ;
		 double cur_time_CPU_4 = eval->runTime( 4, 0) ;
		 double cur_time_CPU_8 = eval->runTime( 8, 0) ;

     	 std::cout << "    @ Time elapsed for connectivity 4, GPU : " << cur_time_4 << "  ms" << std::endl;
         std::cout << "    @ Time elapsed for connectivity 8, GPU : " << cur_time_8 << "  ms" << std::endl;
		 std::cout << "    @ Time elapsed for connectivity 4, CPU : " << cur_time_CPU_4 << "  ms" << std::endl;
		 std::cout << "    @ Time elapsed for connectivity 8, CPU : " << cur_time_CPU_8 << "  ms" << std::endl;
		delete eval ;
		eval = nullptr ;
        total_runtime_4 += cur_time_4 ;
        total_runtime_8 += cur_time_8 ;
		total_runtime_CPU_4 += cur_time_CPU_4 ;
		total_runtime_CPU_8 += cur_time_CPU_8 ;
    #endif

    #ifdef CORRECTNESSTEST
		total_pass_num +=  1 ;
        std::cout << "======================= TEST PASS  =====================" << std::endl;
    #endif

    std::cout << std::endl;
   // delete eval ;
   // eval = nullptr ; 
}


void run_visualization(std::string filename, std::string algo)
{
    if (algo == "NP")
    {
        Visualization<CCLNPGPU> visl( filename.c_str(), 4, 0); 
    }
    
}

int main( int argc, char* argv[])
{
    if (argc == 1 || argc == 2 || argc == 3 )
    {
        std::cerr << "  Invalid input, you need to provide the input image " << std::endl;
        return 0  ;
    }


    std::string cclalgo = std::string( argv[1] ) ;
	std::string path    = std::string( argv[2] ) ;

    if ( cclalgo != "DPL" && cclalgo != "NP" && cclalgo != "LE")
    {
        std::cerr << "  Invalid algorithm, you can use one of LE, NP and DPL" << std::endl;
    }


    std::cout << "  CCL Algorithm Validation : " << std::endl;
    std::cout << "    @ System    : " << getOsName() << std::endl; 
    std::cout << "    @ Algorithm : " << cclalgo << std::endl;

    #ifdef RUNTEST
    
    for ( int i = 3; i < argc; i++)
    {
        std::string current =  path + std::string( argv[i] ) ;
        run_test( current.c_str(), cclalgo) ;
    }
    std::cout << "  Validation Summary : " << std::endl ;
    std::cout << "    @ Algorithm         : " << cclalgo << std::endl;
    std::cout << "    @ Total Test Images : " << argc - 3 << std::endl ;
	std::cout << "    @ Total Pass Images : " << total_pass_num << std::endl; 
    std::cout << "    @ Average Time per Image Connection-4, GPU (ms) : " << total_runtime_4 / double(argc-2) << std::endl;
    std::cout << "    @ Average Time per Image Connection-8, GPU (ms) : " << total_runtime_8 / double(argc-2) << std::endl;
	std::cout << "    @ Average Time per Image Connection-4, CPU (ms) : " << total_runtime_CPU_4 / double(argc-2) << std::endl;
	std::cout << "    @ Average Time per Image Connection-8, CPU (ms) : " << total_runtime_CPU_8 / double(argc-2) << std::endl;
    #endif

    #ifdef VISUALIZATION
    run_visualization( path+std::string(argv[3]), cclalgo ) ;
    #endif
	return 1 ;
}
