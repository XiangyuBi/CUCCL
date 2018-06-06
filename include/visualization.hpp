#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

struct VisualizationCCL
{

    cv::Mat* originImage ;
    cv::Mat* grayImage   ;
    cv::Mat* rgbImage    ;  

    explicit VisualizationCCL( cv::Mat*  )
        : originImage(cv::Mat(cv::imread(filename, cv::IMREAD_GRAYSCALE)))
        {
        }

  
    inline void convertToRGB()
    {
        cv::cvtColor(grayImage, rgbImage, CV_GRAY2RBG) ;
    }; 

    void showImage()
    {
        cv::namedWindow( "Display CCL Result", WINDOW_AUTOSIZE );
        cv::imshow("Display CCL Result", rgbImage) ;
        cv::waitKey(0) ;
    }

};


template < class CCL>
class Visualization
{

private:
    cv::Mat originImage ;
    cv::Mat grayImage   ;
    cv::Mat rgbImage    ;
    CCL algo            ;
    int *label;

public: 
    explicit Visualization( char const* filename, int degreeOfConnectivity, unsigned char threshold  )
        : originImage(cv::Mat(cv::imread(filename, cv::IMREAD_GRAYSCALE)))
    {
        label = new int[ originImage.size().width * originImage.size().height]{0};
        algo.CudaCCL(originImage.data, label, originImage.size().height, originImage.size().width, degreeOfConnectivity, threshold) ;
        grayImage = cv::Mat(image->size().height, image->size().width , CV_32SC1, (void*)label ); 
    }

    ~Visualization()
    {
        delete[] label ;
        label = nullptr ;
    }


    inline void convertToRGB()
    {
        cv::cvtColor(grayImage, rgbImage, CV_GRAY2RBG) ;
    }; 

    void showImage()
    {
        cv::Mat comb( cv::Size( originImage.cols*2 , originImage.rows, originImage.type(), cv::Scalar::all(0) ) ) ;
        cv::Mat roi = comb( cv::Rect(0, 0, originImage.cols, originImage.rows) ) ;
        originImage.copyTo(roi) ;
        roi = comb( cv::Rect( originImage.cols, 0, originImage.cols, originImage.rows ) ) ;
        rgbImage.copyTo(roi) ;

        cv::namedWindow( "Display CCL Result", WINDOW_AUTOSIZE );
        cv::imshow("Display CCL Result", comb) ;
        cv::waitKey(0) ;
    }


    
};




#endif