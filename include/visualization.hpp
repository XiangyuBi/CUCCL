#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>




class VisualizationCCL
{
private:
    cv::Mat grayImage ;
    cv::Mat rgbImage  ;




public: 
    explicit VisualizationCCL(int* gray, size_t height, size_t width)
        : grayImage( cv::Mat(height, width, CV_32SC1, (void*)gray ))
    {
    }

    inline void convertToRGB()
    {
        cv::cvtColor(grayImage, rgbImage, cv::COLOR_GRAY2BGR) ;
    }; 

    void showImage()
    {
        cv::namedWindow( "Display CCL Result",cv::WINDOW_AUTOSIZE );
        cv::imshow("Display CCL Result", rgbImage) ;
        cv::waitKey(0) ;
    }

};





#endif
