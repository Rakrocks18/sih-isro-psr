#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

// CUDA kernel for applying mask and converting image
__global__ void applyMaskKernel(const uchar4* inputImage, const unsigned char* mask, 
                                uchar4* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 pixel = inputImage[idx];
        unsigned char maskValue = mask[idx];
        
        // Apply mask: if mask > 128, keep original alpha, else set to 0
        pixel.w = (maskValue > 128) ? 255 : 0;
        
        outputImage[idx] = pixel;
    }
}

// Host function to set up and launch the CUDA kernel
void applyMaskCUDA(const cv::Mat& inputImage, const cv::Mat& maskImage, cv::Mat& outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    
    // Allocate device memory
    uchar4* d_inputImage;
    unsigned char* d_mask;
    uchar4* d_outputImage;
    
    cudaMalloc(&d_inputImage, width * height * sizeof(uchar4));
    cudaMalloc(&d_mask, width * height * sizeof(unsigned char));
    cudaMalloc(&d_outputImage, width * height * sizeof(uchar4));
    
    // Copy input data to device
    cudaMemcpy(d_inputImage, inputImage.ptr<uchar4>(), width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, maskImage.ptr<unsigned char>(), width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    applyMaskKernel<<<gridSize, blockSize>>>(d_inputImage, d_mask, d_outputImage, width, height);
    
    // Copy result back to host
    cudaMemcpy(outputImage.ptr<uchar4>(), d_outputImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_mask);
    cudaFree(d_outputImage);
}

int main() {
    // Load input image and mask
    cv::Mat inputImage = cv::imread("path/to/input.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat maskImage = cv::imread("path/to/mask.png", cv::IMREAD_GRAYSCALE);
    
    if (inputImage.empty() || maskImage.empty()) {
        printf("Error: Could not read input images\n");
        return -1;
    }
    
    // Ensure the dimensions match
    if (inputImage.size() != maskImage.size()) {
        printf("Error: Input image and mask dimensions do not match\n");
        return -1;
    }
    
    // Create output image
    cv::Mat outputImage(inputImage.size(), CV_8UC4);
    
    // Apply mask using CUDA
    applyMaskCUDA(inputImage, maskImage, outputImage);
    
    // Save output image
    cv::imwrite("path/to/output.png", outputImage);
    
    printf("Converted image saved to path/to/output.png\n");
    
    return 0;
}
