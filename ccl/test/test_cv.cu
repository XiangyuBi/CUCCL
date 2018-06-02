#include "../CUCCL_LE/CUCCL_LE.hpp"
#include "../CUCCL_LE/CUCCL_LE.cuh"
#include "../CUCCL_NP/CUCCL_NP.cuh"
#include "../CUCCL_DPL/CUCCL_DPL.cuh"


#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iomanip>
#include <iostream>

using namespace std; 
using namespace CUCCL; 
using namespace cv;
/*

void testCCL(char const* flag)
{
    const auto width = 32;
	const auto height = 8;
	unsigned char data[width * height] =
	{
		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 135, 120, 120,
		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 135, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120
    };
    
    int labels[width * height] = { 0 };

    cout << "Binary image is : " <<endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << static_cast<int>(data[i * width + j]) << " ";
		}
		cout << endl;
	}
    cout<<endl;
    
    auto degreeOfConnectivity = 4;
    unsigned char threshold = 0;
    
    if (flag == "LE")
    {
        CCLLEGPU ccl;

        ccl.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold) ;
        cout << "Label Mesh by CCL LE : " <<endl;
	    for (auto i = 0; i < height; i++)
	    {
		    for (auto j = 0; j < width; j++)
		    {
			    cout << setw(3) << labels[i * width + j] << " ";
		    }
		    cout << endl;
        }
    }

    if (flag == "NP")
    {
        CCLNPGPU cclnp;
	    cclnp.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold);

	    cout << "Label Mesh by CCL NP : " << endl;
	    for (auto i = 0; i < height; i++)
	    {
		    for (auto j = 0; j < width; j++)
		    {
			    cout << setw(3) << labels[i * width + j] << " ";
		    }
		    cout << endl;
	    }
	}
	
	if (flag == "DPL")
	{
		CCLDPLGPU ccldpl;
		ccldpl.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold);

		cout << "Label Mesh by CCL DPL : " << endl;
		for (auto i = 0; i < height; i++)
		{
			for (auto j = 0; j < width; j++)
			{
				cout << setw(3) << labels[i * width + j] << " ";
			}
			cout << endl;
		}

	}
}

*/

void test_image( char const* path, char const* flag)
{
	Mat* image ;
	image = new Mat(imread(path, IMREAD_GRAYSCALE));
	int* label = new int[image->size().width * image->size().height]{0} ;
	unsigned char* data  = image->data ;
	auto degreeOfConnectivity = 4 ;
	unsigned char threshold = 0 ;
	if (flag == "DPL")
	{
		CCLDPLGPU ccldpl;
		ccldpl.CudaCCL(data, label, image->size().height, image->size().width, degreeOfConnectivity, threshold);
	}
	namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
	Mat* result = new Mat(image->size().height, image->size().width , CV_32SC1, (void*)label );
	imshow( "Display window", *result );                // Show our image inside it.
	
	waitKey(0); // Wait for a keystroke in the window
	delete image ;
	delete label ;
	delete result ;
	image = nullptr ;
	label = nullptr ;
	result = nullptr ;
}
int main()
{
    test_image("test.png", "DPL");
	return 1 ;

}
