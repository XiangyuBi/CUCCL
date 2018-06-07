#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <iostream>

#include <ctime>
#include <chrono>
#include <string>

#include <vector>
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
    int    runCorrectness(int cpuLabel) ;


    

}; 

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
