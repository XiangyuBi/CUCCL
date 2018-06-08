#ifndef CUCCL_DPL_CUH
#define CUCCL_DPL_CUH
#include <host_defines.h>


namespace CUCCL{

__global__ void init_CCLDPL(int *gLabel, int width, int height);
    
__global__ void dpl_kernel_4(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre);
    
__global__ void dpl_kernel_8(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre);
    
class CCLDPLGPU
{
public:
    explicit CCLDPLGPU(unsigned char* dataOnDevice = nullptr, int* labelListOnDevice = nullptr)
            : FrameDataOnDevice(dataOnDevice),
              LabelListOnDevice(labelListOnDevice)
    {
    }
    
    void CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold);
    
private:
    unsigned char* FrameDataOnDevice;
    int* LabelListOnDevice;
};
}


#endif