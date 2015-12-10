#ifndef __BLUR_DETECT_H__
#define __BLUR_DETECT_H__
#pragma GCC diagnostic ignored "-Wunused-value"

#include <opencv2/opencv.hpp>

#include "Deconvolution.h"

namespace BRT {

namespace IMG_STATISTICS {

inline double getMean(const cv::Mat img, const cv::Mat mask=cv::Mat()) {
    assert(("ERROR In getMean: Image should be single channel.", img.channels() == 1));
    return cv::mean(img, mask).val[0];
}

inline double getVariance(const cv::Mat img, const cv::Mat mask=cv::Mat()) {
    assert(("ERROR In getVariance: Image should be single channel.", img.channels() == 1));
    cv::Scalar mu, sigma;
    cv::meanStdDev(img, mu, sigma, mask);
    return sigma.val[0];
}

inline double getSquareOfVariance(const cv::Mat img, const cv::Mat mask=cv::Mat()) {
    assert(("ERROR In getSquareOfVariance: Image should be single channel.", img.channels() == 1));
    cv::Scalar mu, sigma;
    cv::meanStdDev(img, mu, sigma, mask);

    return (sigma.val[0]*sigma.val[0]);
}

inline double getNormalizedGraylevelVariance(const cv::Mat img, const cv::Mat mask=cv::Mat()) {
    assert(("ERROR In getNormalizedGraylevelVariance: Image should be single channel.", img.channels() == 1));
    cv::Scalar mu, sigma;
    cv::meanStdDev(img, mu, sigma, mask);

    return (sigma.val[0]*sigma.val[0]) / mu.val[0];
}

inline cv::Mat getDensityDistributionMask(cv::Mat img, double sigma, double mean, double boxRange) {
    cv::Mat rtn;
    assert(("ERROR In getDensityDistributionMask: Image should be single channel.", img.channels() == 1));
    cv::inRange((img-mean), -boxRange*sigma, boxRange*sigma, rtn);
    return rtn;
}

}
}

/**
 * @brief The BlurDetector class
 *
 * Main driver class for performing blurness detection, There are six different methods in its arsenal
 * that can be used to detect bluriness in the image. Default method used is variance of Gaussian.
 *
 * Expects an input image of type cv::Mat and provides blurness measure as a double.
 *
 */
class BlurDetector
{
public:
    /**
     * @brief BlurDetector
     *  Default Constructor - Does nothing special.
     */
    BlurDetector();

    /**
     * @brief The BLUR_DETECTION_TYPE enum
     * Different methods of computing bluriness in the image.
     */
    enum BLUR_DETECTION_TYPE{
        LAPM,   // Modified Laplacian
        LAPV,   // Variance of Laplacian
        GAUSSV, // Variance of Gaussian
        GRAD,   // Sobel Gradients
        MEDV,   // Variance of Median
        DEBLUR  // Deblurring
    };

    /**
     * @brief detectBluriness
     * @param img
     * @param detectionAlgorithm
     * @return double blurMeasure
     *
     * Main wrapper function that is exposed to detect bluriness in the image.
     */
    double detectBluriness(cv::Mat img,
                           int detectionAlgorithm=GAUSSV);

private:
    /**
     * @brief preprocessImage
     * @param img
     * @param mask
     * @return img
     *
     * Preprocessing step makes the image data as a zero-mean unit variance data.
     */
    cv::Mat preprocessImage(const cv::Mat img, const cv::Mat mask);

    /**
     * @brief modifiedLaplacian
     * @param src
     * @return double FocusMeasure
     *
     * Seperable dX and dY operators, seperable filters for column index and row index,
     * dX will be operated by second order derivative and dY will be operated
     * by gaussian kernel derivative.
     *
     * Returns the mean value of absolute sum of seperable dX and dY.
     */
    double modifiedLaplacian(const cv::Mat src);

    /**
     * @brief varianceOfLaplacian
     * @param src
     * @return double - Focus Measure
     *
     * Computes square of the variance of second order derivative(laplacian) filter.
     */
    double varianceOfLaplacian(const cv::Mat src);

    /**
     * @brief varianceOfGaussian
     * @param src
     * @return double blurMeasure
     *
     * Computes the square of variance on difference between source image and processed image,
     * gives an estimate of how much of blur is seen in the image. Close to zero represents image
     * is very blurry. Uses a gaussian kernel of variable size based on input image.
     */
    double varianceOfGaussian(const cv::Mat src);

    /**
     * @brief varianceOfMedian
     * @param src image
     * @return double blurMeasure
     *
     * Computes the square of variance on difference between source image and processed image,
     * gives an estimate of how much of blur is seen in the image. Close to zero represents image
     * is very blurry. Uses a median filter of variable size based on input image.
     */
    double varianceOfMedian(const cv::Mat src);

    /**
     * @brief sobelGradients
     * @param src
     * @param ksize
     * @return double focusMeasure
     *
     * Computes the sobel derivative along dX and dY, computes the mean on the sum of
     * square of the gradients.
     */
    double sobelGradients(const cv::Mat src, int ksize);

    /**
     * @brief deblurResponse
     * @param img
     * @return double blurMeasure
     *
     * Perfoms deblurring of the image using wiener filter approximation, Measure is inversely
     * proportional to the amount of stress it takes to deblur the image. More the stress lesser
     * the blurMeasure.
     */
    double deblurResponse(const cv::Mat img);

    /**
     * @brief maskSaturatedPixels
     * @param img
     * @return img - masked non-saturated regions in image.
     *
     * Masks the saturated image pixels in the input image and returns the image mask.
     */
    cv::Mat maskSaturatedPixels(const cv::Mat img);

    Deconvolution* _deconv;
};

#endif
