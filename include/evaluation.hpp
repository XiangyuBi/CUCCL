#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <iostream>
#include <fstream>

#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


template <class CCL>
class Evaluation{
private:
    CCL algo ;
    cv::Mat image ;
    int *label ;
    std::string image_name ;

public:
    explicit Evaluation(char const* filename)
        : image_name(std::string(filename))
        , image(cv::Mat(cv::imread(filename, cv::IMREAD_GRAYSCALE)))
        {
            label = new int[image.size().width * image.size().height]{0} ;
        } ;
    
    ~Evaluation()
    {
        delete[] label ;
        label = nullptr;
    }

    double runTime(int degreeOfConnectivity, unsigned char threshold);
    void   runOutput( char const* path) ;
    int    runCorrectness(int cpuLabel) ;



    

}; 


template<class CCL>
void Evaluation<CCL>::runOutput(std::string path)
{
    auto* indexes = new std::vector< std::vector<int> >() ;
    std::unordered_map<int, int> the_labels ;
    int label_num = 0 ;
    
    for( int i = 0; i < image.cols; i++)
    {
        for (int j = 0; j< image.rows; j++)
        {
            int cur_value = label[ i * image.rows + j] ;
            if ( the_labels.find( cur_value ) != the_labels.end() )
                (*indexes)[ the_labels[cur_value] ].push_back( i * image.rows + j) ;
            else
            {
                the_labels[ cur_value ] = label_num ;
                indexes->push_back( { i * image.rows + j} ) ;
                label_num ++ ;
            }
            
        }
    }
    std::string outputname = path + image_name + ".txt" ;
    std::ofstream output ;
    output.open(outputname.c_str()) ;

    for(int m = 0; m < indexes->size(); m++)
    {
        auto cur_vec = (*indexes)[m] ;
        for ( int n = 0; n < cur_vec.size(); n++)
        {
            output << cur_vec[n] << " " ;
        }
        output << std::endl ;
    }
    output.close() ;
    delete indexes ;
    indexes = nullptr ;  

}



template <class CCL>
double Evaluation<CCL>::runTime(int degreeOfConnectivity, unsigned char threshold)
{
    
    auto t_start = std::chrono::high_resolution_clock::now();

    algo.CudaCCL(image.data, label, image.size().height, image.size().width, degreeOfConnectivity, threshold ) ;

    auto t_end = std::chrono::high_resolution_clock::now();

    double elaspedTimeMs = std::chrono::duration<double, std::milli>(t_end-t_start).count();
}

template <class CCL>
int Evaluation<CCL>::runCorrectness(int cpuLabel)
{

    int * curLabel = label ;
    std::vector<int> values ;
    return 1 ;
    for(auto i = 0; i < image.size().height * image.size().width ; i ++)
    {
        if ( std::find( values.begin(), values.end(), curLabel[i] ) == values.end())
            values.push_back( curLabel[i] ) ;
    }
    
    return (cpuLabel == values.size() ) ; 

}

#endif
