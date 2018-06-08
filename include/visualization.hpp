#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <stdlib.h>

template < class CCL>
class Visualization
{

private:
    cv::Mat originImage ;
    cv::Mat* grayImage   ;
    cv::Mat rgbImage    ;
    CCL algo            ;
    int *label ;

	int random_color = 10 ;
	std::vector<unsigned char> color_r ;
	std::vector<unsigned char> color_g ;
	std::vector<unsigned char> color_b ;

public: 
    explicit Visualization( char const* filename, int degreeOfConnectivity, unsigned char threshold  )
        : originImage(cv::Mat(cv::imread(filename, cv::IMREAD_GRAYSCALE)))
    
	{
		std::cout << "  Visualization CCL :" << std::endl;
		std::cout << "    @ Filename : " << filename << std::endl; 
        label = new int[ originImage.size().width * originImage.size().height]{0};
        algo.CudaCCL(originImage.data, label, originImage.size().width, originImage.size().height, degreeOfConnectivity, threshold) ;
     //   grayImage = cv::Mat(originImage.cols, originImage.rows , CV_32SC1, (void*)label ); 
		generate_color() ;
		convertToRGB()	;
		showImage() ;
    }

    ~Visualization()
    {
        delete[] label ;
        label = nullptr ;
		delete grayImage ;
		grayImage = nullptr ;
    }

	
	void generate_color()
	{
		srand(time(NULL));
		for ( int i = 0; i < random_color ; i ++)
		{
			unsigned char r = rand() % 255 ;
			unsigned char g = rand() % 255 ;
			unsigned char b = rand() % 255 ;
			color_r.push_back(r) ;
			color_g.push_back(g) ;
			color_b.push_back(b) ;
		}
	}

	void map_to_rgb()
	{
		for(int y=0;y<rgbImage.rows;y++)
		{
			    for(int x=0;x<rgbImage.cols;x++)
				 {
				       cv::Vec3b color = rgbImage.at<cv::Vec3b>(cv::Point(x,y));
					   if ( color[0] == 0 )
					   {
						   color[0] = 255 ;
						   color[1] = 255 ;
						   color[2] = 255 ;
						   rgbImage.at<cv::Vec3b>(cv::Point(x,y)) = color ;	
						   continue ;
					   }
					   
						
					   int ind = color[0] % random_color ;
					   color[0] = color_r[ind];
					   color[1] = color_g[ind];
					   color[2] = color_b[ind];
					   rgbImage.at<cv::Vec3b>(cv::Point(x,y)) = color;		
				 }
		}
	}

	

    inline void convertToRGB()
    {
        for ( int i = 0; i < originImage.cols * originImage.rows ; i++)
			label[i] = label[i] % 256 ; 
		grayImage = new cv::Mat( originImage.rows, originImage.cols,  CV_32SC1, (void*)label) ;  // grayImage->convertTo( *grayImage, CV_8UC1, 1 / 255.0);
		//grayImage->convertTo( *grayImage, CV_32FC1);
		cv::Mat temp ;
		grayImage->convertTo( temp, CV_8UC1, 1 ) ;
		cv::cvtColor(temp, rgbImage, cv::COLOR_GRAY2BGR, 3 ) ;
		cv::cvtColor(originImage, originImage, cv::COLOR_GRAY2BGR, 3) ;
   		map_to_rgb() ;
	}; 

    void showImage()
    {
        cv::Mat comb( cv::Size( originImage.cols*2 , originImage.rows), originImage.type(), cv::Scalar::all(0)  ) ;
        cv::Mat roi = comb( cv::Rect(0, 0, originImage.cols, originImage.rows) ) ;
        originImage.copyTo(roi) ;
        roi = comb( cv::Rect( originImage.cols, 0, originImage.cols, originImage.rows ) ) ;
        rgbImage.copyTo(roi) ;

        cv::namedWindow( "Display CCL Result", cv::WINDOW_AUTOSIZE );
        cv::imshow("Display CCL Result", comb) ;
        cv::waitKey(0) ;
    }


    
};




#endif
