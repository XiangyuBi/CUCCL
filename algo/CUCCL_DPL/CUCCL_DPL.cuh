#ifndef CUCCL_DPL_CUH
#define CUCCL_DPL_CUH
#include <host_defines.h>


namespace CUCCL{

void init_label_list(int *labelList, int width, int height);

void set_kernel_dim(int width, int height, dim3 &block, dim3 &grid);

void print_init_labels(int width, int height, int* labels);
    
__global__ void dpl_kernel_4(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre);
    
__global__ void dpl_kernel_8(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre);
    
class CCLDPLGPU
{
public:
    explicit CCLDPLGPU(unsigned char* dataOnDevice = nullptr, int* labelListOnDevice = nullptr)
            : gData(dataOnDevice),
              gLabelList(labelListOnDevice)
    {
    }
    
    void CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold);
    
private:
    unsigned char* gData;
    int* gLabelList;
};
}


#endif