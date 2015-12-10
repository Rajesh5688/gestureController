#ifndef DEBLURCONTEXT_H
#define DEBLURCONTEXT_H

#include <opencv2/opencv.hpp>
#include "blur_modes/Blur.h"

/**
 * @brief The DeconvContext class
 *
 * This class acts as a message handler for ease of passing messages around
 * the deconvolution function calls.
 */
class DeconvContext {

public:
    cv::Mat _inputImage;
    cv::Mat _inputImageMatrix;
    cv::Mat _outputImageMatrix;
    cv::Mat _inputImageFFT;
    cv::Mat _outputImageFFT;
    cv::Mat _kernelFFT;
    int _width;
    int _height;
    Blur *_blur;
};

#endif
