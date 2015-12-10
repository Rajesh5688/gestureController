#ifndef DECONVOLUTION_H
#define DECONVOLUTION_H
#include <QDebug>
#include <QTime>
#include <QPainter>
#include <time.h>
#include <math.h>

#include "blur_modes/Blur.h"
#include "blur_modes/FocusBlur.h"
#include "blur_modes/GaussianBlur.h"
#include "blur_modes/MotionBlur.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/any.hpp>
#include <map>

#include <opencv2/opencv.hpp>
#include "DeblurContext.h"

/**
 * @brief The Deconvolution class
 *
 * Driver class that perfoms image deblurring operation, Handles two different methods
 * to perform image deblurring. (1) Weiner Filter Approximation(Default), (2) Tikhonov Regularization
 *
 * NOTE: Does not handle fully saturated images.
 */
class Deconvolution
{
public:
    /**
     * @brief Deconvolution
     *
     * Initializes the low frequency blur kernel to perform deconvolution. Sets up the DFT plan for both forward and
     * inverse DFT's.
     */
    Deconvolution();

    /**
      * Calls removeFFT to clear al the allocated spaces.
      */
    ~Deconvolution();

    /**
     * @brief doDeconvolution
     * @param inputImage
     * @param  reference to outputImage
     * @return bool that says if convolution was successful. No real use so far.
     *
     * Main handler function that performs deConvolution, assigns DeconvContext and reads number of channels
     * present in the image and performs deconvolution on each channel.
     */
    bool doDeconvolution(cv::Mat inputImage, cv::Mat& outputImage);

    /**
     * @brief visualizeFFT
     * @param fft
     * @param path
     *
     * Displays the magnitude of the frequency spectrum, Spectrum output will be centered with respect
     * to the center of the image.
     */
    void visualizeFFT(cv::Mat fft, QString path);

    /**
     * @brief initFFT
     * @param img
     *
     * Initializes matrices and array dimensions to be used in fourier transformations.
     */
    void initFFT(const cv::Mat img);

private:
    /**
     * @brief buildKernel
     * @param pass by reference outKernelFFT
     * @param width
     * @param height
     *
     * Builds a kernel dimension in frequency that matches image dimension.
     * Uses the kernel that got created in constructor.
     * Maintains the low frequcny specs of filter kernel.
     */
    void buildKernel(cv::Mat& outKernelFFT, const int width, const int height);

    /**
     * @brief forwardDft
     * @param inputType
     * @return bool success/failed
     *
     * Perform forward DFT, uses input type specified and calls the respective spatial and frequency domain matrices
     * as specified by dftPlan.
     */
    bool forwardDft(std::string inputType);

    /**
     * @brief inverseDft
     * @param inputType
     * @return bool success/failed
     *
     * Perform inverse DFT, uses input type specified and calls the respective spatial and frequency domain matrices
     * as specified by idftPlan.
     */
    bool inverseDft(std::string inputType);

    /**
     * @brief removeFFTObjects
     *
     * Deallocate matrices used to hold data of frequency domain.
     */
    void removeFFTObjects();

    /**
     * @brief pass by reference multiplyRealFFTs
     * @param outFFT
     * @param _kernelFFT
     * @param width
     * @param height
     *
     * Multiply matrices of real part of frequency domain matrices.
     */
    static void multiplyRealFFTs(cv::Mat& outFFT, const cv::Mat _kernelFFT,
                                 const int width, const int height);

    /**
     * @brief deconvolutionByWiener
     * @param processingContext
     *
     * Performs deconvolution as per Weiner Deconvolution Algorithm. Algorithm basically approximates a particular
     * image to deblur by dividing the input image in the frequency domain with a low pass frequency kernel in the freqeuncy domain.
     */
    void deconvolutionByWiener(DeconvContext* processingContext);

    /**
     * @brief deconvolutionByTikhonov
     * @param processingContext
     *
     * Performs deconvolution as per Tikhonov Deconvolution Algorithm. Algorithm approximates image to deblur by taking out the
     * energy value of low frequency kernel from the input image. Also models noise in the image as a second order function. Which
     * does better in reducing deblur noise artifacts been added to deblured image.
     */
    void deconvolutionByTikhonov(DeconvContext *processingContext);

    /**
     * @brief doDeconvolutionForChannel
     * @param processingContext
     *
     * Wrapper function that handles foward and inverse DFT plans and also calls specific Deconvolution method to run.
     */
    void doDeconvolutionForChannel(DeconvContext* processingContext);

    /**
     * @brief getImageFromMatrix
     * @param processingContext
     * @return Output Image from matrix
     *
     * As function name specifies, function takes the output matrix generated from inverse DFT and converts it to a uint8 image.
     */
    cv::Mat getImageFromMatrix(DeconvContext* processingContext);

    // Might be a good option to use this flag if using this function seperately for multi-threading.
    volatile bool _isProcessingCancelled;
    int _width, _height;
    /** TODO : Currently not been used, always uses Wiener Approach,
     *         If set to anything other than 0 uses tikhonov approach. **/
    int _deblurMethod;

    // Boost Function to bind connections.
    boost::function<bool (std::string) > _dftPlan;
    boost::function<bool (std::string) > _idftPlan;

    // OpenCV Usages -- Spatial domain data
    cv::Mat _inputImageMatrix;
    cv::Mat _outputImageMatrix;
    cv::Mat _kernelMatrix;
    cv::Mat _laplacianMatrix;
    cv::Mat _outLaplacianMatrix;

    // Opencv Usages -- freuquency domain data
    cv::Mat _inputImageFFT;
    cv::Mat _outputImageFFT;
    cv::Mat _kernelFFT;
    cv::Mat _kernelTempFFT;
    cv::Mat _laplacianMatrixFFT;

    std::map<std::string, cv::Mat> _spatialMap;
    std::map<std::string, cv::Mat> _frequencyMap;

    Blur* _blur;
};
#endif // DECONVOLUTION_H
